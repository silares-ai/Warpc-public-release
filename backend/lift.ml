(* The lifter: warpc's SASS text to PTX, for ptxas to encode. It reads the
   text alone -- the header and one instruction a line, as warpc prints them
   -- and writes each instruction as the PTX that asks for the same work.

   Most instructions are one PTX instruction each. What differs:
   - An instruction of the uniform datapath that issues work once for the
     warp -- an MMA, a commit, a copy-engine load or store, a tensor-map
     prefetch, a request to the scheduler, a barrier's initialisation -- is a
     thread's instruction in PTX, so it runs on the one lane elect.sync
     picks at the kernel's entry: a predicate ptxas knows is one lane, where
     a lane-number test costs a loop over the lanes it might be. Its uniform
     guard is a branch around it, which ptxas lifts out of the loop it sits
     in; a guard joined to the lane's predicate is a vote before each. A bulk
     group's commit and wait are each thread's own in PTX: the warp's are
     every lane's, the lanes that issued nothing committing and waiting for
     nothing.
   - Tensor memory's allocation is a construct in the SASS -- an elected
     lane asks the allocator until it answers, the warp meets after -- and
     one instruction in PTX, whose answer lands in shared memory past the
     kernel's own; its free names a mask in the SASS and an address and a
     column count in PTX, decoded where it runs.
   - What PTX asks for that has no instruction of its own: a wait for
     tensor-memory loads before their registers are used, the fence that
     publishes the barriers' initialisation to the cluster, the tensor-core
     fences around a thread synchronisation. The cluster's arrival is the
     relaxed one: the SASS's orders no memory of its own.
   - A try-wait's parity is the register's bit 31 in the SASS and bit 0 in
     PTX, and its sleep while the phase is not complete is PTX's
     suspend-time hint; a barrier's arrival count is decoded from its
     initialisation word.
   - The SASS addresses shared memory from 0x400, where the kernel's window
     starts: the lifter declares the window and the kernel traps if ptxas
     put it anywhere else, so every address stays what warpc derived.
   - The SASS's registers are allocated: one register holds many values in
     turn. Each value's live range -- the definitions that reach a common
     read -- becomes a register of its own, as a compiler hands ptxas its
     virtual registers. The kernel states the CTAs an SM holds by its shared
     memory, the occupancy ptxas allocates those registers for.

   Anything else is refused. *)

let pf = Printf.sprintf
let fail fmt = Printf.ksprintf failwith ("Lift: " ^^ fmt)
let starts p s = String.length s >= String.length p && String.sub s 0 (String.length p) = p

let contains sub s =
  let n = String.length sub and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = sub || go (i + 1)) in
  go 0

(* ---- the text ---- *)

type header =
  { name : string
  ; threads : int
  ; smem : int (* shared memory from 0x400: the static block and the dynamic *)
  ; params : int
  ; cluster : int
  ; sm : string (* the PTX target: sm_100a, the B200, or sm_103a, the B300 *)
  }

(* an operand: a register, an immediate, a name *)
type opnd =
  | R of int (* RZ is R 255 *)
  | U of int (* URZ is U 63 *)
  | P of bool * int (* negated, number; PT is 7 *)
  | Q of bool * int (* uniform predicates; UPT is 7 *)
  | Imm of int
  | Name of string (* a label, a special register, SB0 *)

type line =
  | Lab of string
  | Ins of { guard : opnd option; op : string; args : string list; text : string }

let strip_ctrl l = match String.index_opt l '{' with Some i -> String.trim (String.sub l 0 i) | None -> String.trim l

(* operands split at top-level commas *)
let split_args s =
  let out = ref [] and buf = Buffer.create 16 and depth = ref 0 in
  String.iter
    (fun c ->
      match c with
      | '[' -> incr depth; Buffer.add_char buf c
      | ']' -> decr depth; Buffer.add_char buf c
      | ',' when !depth = 0 -> out := String.trim (Buffer.contents buf) :: !out; Buffer.clear buf
      | c -> Buffer.add_char buf c)
    s;
  if Buffer.length buf > 0 then out := String.trim (Buffer.contents buf) :: !out;
  List.rev !out

let int_of s =
  let s = String.trim s in
  if starts "-" s then - int_of_string (String.sub s 1 (String.length s - 1)) else int_of_string s

let num p s = int_of_string (String.sub s (String.length p) (String.length s - String.length p))

let reg s =
  let s = String.trim s in
  let neg, s = if starts "!" s then true, String.sub s 1 (String.length s - 1) else false, s in
  match s with
  | "RZ" -> R 255
  | "URZ" -> U 63
  | "PT" -> P (neg, 7)
  | "UPT" -> Q (neg, 7)
  | _ when starts "UR" s -> U (num "UR" s)
  | _ when starts "UP" s -> Q (neg, num "UP" s)
  | _ when starts "R" s && String.length s > 1 && s.[1] >= '0' && s.[1] <= '9' -> R (num "R" s)
  | _ when starts "P" s && String.length s > 1 && s.[1] >= '0' && s.[1] <= '9' -> P (neg, num "P" s)
  | _ when starts "0x" s || starts "-0x" s -> Imm (int_of s)
  | _ -> Name s

let parse_line l =
  let l = strip_ctrl l in
  if l = "" || l.[0] = '#' || l.[0] = '.' then None
  else if l.[String.length l - 1] = ':' then Some (Lab (String.sub l 0 (String.length l - 1)))
  else begin
    let guard, rest =
      if l.[0] = '@' then
        let i = String.index l ' ' in
        Some (reg (String.sub l 1 (i - 1))), String.trim (String.sub l i (String.length l - i))
      else None, l
    in
    let op, args = match String.index_opt rest ' ' with Some i -> String.sub rest 0 i, String.sub rest (i + 1) (String.length rest - i - 1) | None -> rest, "" in
    Some (Ins { guard; op; args = (if args = "" then [] else split_args args); text = l })
  end

let parse_header lines =
  let get k = List.find_map (fun l -> let w = String.split_on_char ' ' l in match w with x :: v :: _ when x = k -> Some v | _ -> None) lines in
  let need k = match get k with Some v -> v | None -> fail "the header has no %s" k in
  let params = match List.find_opt (starts ".params ") lines with Some l -> List.length (String.split_on_char ' ' l) - 1 | None -> 0 in
  { name = need ".kernel"; threads = int_of (need ".threads"); smem = (match get ".smem" with Some v -> int_of v | None -> 0) + int_of (need ".dynsmem"); params
  ; cluster = (match get ".cluster" with Some v -> int_of v | None -> 1)
  ; sm = (match get ".sm" with Some v -> v | None -> "sm_100a") }

(* [UR4+0x10] -> (base, offset) *)
let addr s =
  let s = String.trim s in
  if not (starts "[" s) then fail "address %s" s;
  let inner = String.sub s 1 (String.length s - 2) in
  match String.index_opt inner '+' with
  | Some i -> String.sub inner 0 i, int_of (String.sub inner (i + 1) (String.length inner - i - 1))
  | None -> inner, 0

(* desc[UR4][R2.64+0x10] -> the pair's first register and the offset *)
let gaddr s =
  match String.index_opt s ']' with
  | Some i ->
    let rest = String.sub s (i + 1) (String.length s - i - 1) in
    let base, off = addr rest in
    if not (Filename.check_suffix base ".64") then fail "global address %s" s;
    num "R" (Filename.chop_suffix base ".64"), off
  | None -> fail "global address %s" s

(* inside name[...] *)
let bracket prefix s =
  if not (starts (prefix ^ "[") s) then fail "%s operand %s" prefix s;
  String.sub s (String.length prefix + 1) (String.length s - String.length prefix - 2)

(* ---- registers: a SASS register, and the PTX register each read or write
   of it names ---- *)

type key =
  | KR of int
  | KU of int
  | KP of int
  | KQ of int

(* how an instruction names its registers: a read, a write *)
type names =
  { use : key -> string
  ; def : key -> string
  }

(* ---- the lift of one instruction ---- *)

type state =
  { out : Buffer.t
  ; mutable pending : int list (* registers a tensor-memory load writes, not yet waited for *)
  ; mutable inits : bool (* a barrier was initialised: the cluster's arrival publishes it *)
  ; mutable tmp : int
  ; mutable skips : int (* the branches around guarded single-issue work *)
  ; mutable skip : string option (* the label after this instruction *)
  ; grp : int (* the .cta_group of every tcgen05 instruction: 2 for a kernel that allocates for the CTA pair *)
  }

let emit st fmt = Printf.ksprintf (fun s -> Buffer.add_string st.out ("\t" ^ s ^ "\n")) fmt
let wait_ld st = if st.pending <> [] then (emit st "tcgen05.wait::ld.sync.aligned;"; st.pending <- [])

(* the general registers an instruction mentions, read or written, for the
   wait on a tensor-memory load *)
let mentioned args =
  List.concat_map
    (fun a ->
      let a = String.trim a in
      let rec scan i acc =
        if i >= String.length a then acc
        else if a.[i] = 'R' && (i = 0 || a.[i - 1] <> 'U') && i + 1 < String.length a && a.[i + 1] >= '0' && a.[i + 1] <= '9' then begin
          let j = ref (i + 1) in
          while !j < String.length a && a.[!j] >= '0' && a.[!j] <= '9' do incr j done;
          let n = int_of_string (String.sub a (i + 1) (!j - i - 1)) in
          let wide = !j + 3 <= String.length a && String.sub a !j 3 = ".64" in
          scan !j (if wide then n :: (n + 1) :: acc else n :: acc)
        end
        else scan (i + 1) acc
      in
      scan 0 [])
    args

let cmp_of op = match String.split_on_char '.' op with _ :: c :: _ -> String.lowercase_ascii c | _ -> fail "compare %s" op

let lift_ins (nm : names) st guard op args text =
  (* reads and writes by the SASS register's number *)
  let ru n = if n = 255 then "0" else nm.use (KR n) and rd n = nm.def (KR n) in
  let uu n = if n = 63 then "0" else nm.use (KU n) and ud n = nm.def (KU n) in
  let pu = function
    | P (_, 7) | Q (_, 7) -> "%ptrue"
    | P (_, n) -> nm.use (KP n)
    | Q (_, n) -> nm.use (KQ n)
    | _ -> fail "%s: not a predicate" text
  in
  let pd = function P (_, n) when n <> 7 -> nm.def (KP n) | Q (_, n) when n <> 7 -> nm.def (KQ n) | _ -> fail "%s: not a predicate" text in
  let pneg = function P (b, _) | Q (b, _) -> b | _ -> false in
  let src s = match reg s with R n -> ru n | U n -> uu n | Imm v -> string_of_int v | _ -> fail "%s: operand %s" text s in
  let dst s = match reg s with R n when n <> 255 -> rd n | U n when n <> 63 -> ud n | _ -> fail "%s: destination %s" text s in
  let rnum s = match reg s with R n -> n | U n -> n | _ -> fail "%s: register %s" text s in
  let a k = match List.nth_opt args k with Some x -> x | None -> fail "%s: operand %d" text k in
  (* a register pair read as a 64-bit value *)
  let pair f n = st.tmp <- (st.tmp + 1) mod 8; let d = pf "%%d%d" st.tmp in emit st "mov.b64 %s, {%s, %s};" d (f n) (f (n + 1)); d in
  let g = match guard with None -> "" | Some gd -> pf "@%s%s " (if pneg gd then "!" else "") (pu gd) in
  (* one lane: the lane the kernel's entry elected in each warp. A uniform
     guard is a branch around the instruction, which ptxas issues unguarded
     from the warp; a lane's guard and the elected lane are one predicate *)
  let one_lane () =
    match guard with
    | None -> "@%pe "
    | Some (Q _ as gd) ->
      st.skips <- st.skips + 1;
      let l = pf "LIFT_SKIP_%d" st.skips in
      emit st "@%s%s bra %s;" (if pneg gd then "" else "!") (pu gd) l;
      st.skip <- Some l;
      "@%pe "
    | Some gd -> emit st "and.pred %%pg, %s%s, %%pe;" (if pneg gd then "!" else "") (pu gd); "@%pg "
  in
  (* any use of a register a tensor-memory load is still filling waits for the loads *)
  if List.exists (fun x -> List.mem x st.pending) (mentioned args) then wait_ld st;
  let mov d s = emit st "%smov.b32 %s, %s;" g d s in
  (match op with
  (* ---- the integer and uniform ALUs ---- *)
  | "MOV" | "R2UR" | "UMOV" -> mov (dst (a 0)) (src (a 1))
  | "IMAD.MOV.U32" | "IMAD.U32" -> mov (dst (a 0)) (src (a 3))
  | "IMAD" | "UIMAD" ->
    if a 3 = "RZ" || a 3 = "URZ" then emit st "%smul.lo.u32 %s, %s, %s;" g (dst (a 0)) (src (a 1)) (src (a 2))
    else emit st "%smad.lo.u32 %s, %s, %s, %s;" g (dst (a 0)) (src (a 1)) (src (a 2)) (src (a 3))
  | "IMAD.WIDE.U32" ->
    let c = pair ru (rnum (a 3)) in
    emit st "%smad.wide.u32 %s, %s, %s, %s;" g c (src (a 1)) (src (a 2)) c;
    let d = rnum (a 0) in
    emit st "%smov.b64 {%s, %s}, %s;" g (rd d) (rd (d + 1)) c
  | "IADD3" | "UIADD3" ->
    let x = a 3 and y = a 4 in
    if starts "-" y then emit st "%ssub.u32 %s, %s, %s;" g (dst (a 0)) (src x) (src (String.sub y 1 (String.length y - 1)))
    else emit st "%sadd.u32 %s, %s, %s;" g (dst (a 0)) (src x) (src y)
  | "LOP3.LUT" | "ULOP3.LUT" ->
    (* a function of a and b alone is PTX's own instruction for it: ptxas
       12.8 and 12.9 crash on a lop3.b32 whose result is tested equal to 0
       under a guard, and make the same LOP3 of the others *)
    let lut = int_of (a 4) in
    let f = List.map (fun b -> (lut lsr b) land 1) [ 0; 2; 4; 6 ] in
    let d = dst (a 0) and x = src (a 1) and y = src (a 2) in
    (match (a 3 = "RZ" || a 3 = "URZ"), f with
     | true, [ 0; 0; 0; 1 ] -> emit st "%sand.b32 %s, %s, %s;" g d x y
     | true, [ 0; 1; 1; 1 ] -> emit st "%sor.b32 %s, %s, %s;" g d x y
     | true, [ 0; 1; 1; 0 ] -> emit st "%sxor.b32 %s, %s, %s;" g d x y
     | true, [ 1; 0; 1; 0 ] -> emit st "%snot.b32 %s, %s;" g d y
     | true, [ 1; 1; 0; 0 ] -> emit st "%snot.b32 %s, %s;" g d x
     | _ -> emit st "%slop3.b32 %s, %s, %s, %s, %s;" g d x y (src (a 3)) (a 4))
  | "SHF.R.U32.HI" | "USHF.R.U32.HI" -> emit st "%sshr.u32 %s, %s, %s;" g (dst (a 0)) (src (a 3)) (src (a 2))
  | "USHF.L.U32" -> emit st "%sshl.b32 %s, %s, %s;" g (dst (a 0)) (src (a 1)) (src (a 2))
  | "LEA" | "ULEA" ->
    emit st "%sshl.b32 %%t1, %s, %s;" g (src (a 1)) (src (a 3));
    emit st "%sadd.u32 %s, %%t1, %s;" g (dst (a 0)) (src (a 2))
  | "FADD" -> emit st "%sadd.rn.f32 %s, %s, %s;" g (dst (a 0)) (src (a 1)) (src (a 2))
  | _ when starts "ISETP." op || starts "UISETP." op -> emit st "%ssetp.%s.u32 %s, %s, %s;" g (cmp_of op) (pd (reg (a 0))) (src (a 2)) (src (a 3))
  | "PLOP3.LUT" ->
    (* P = UP: the form the programs use *)
    if not (a 1 = "PT" && a 2 = "PT" && a 3 = "PT" && a 5 = "0x80" && a 6 = "0x8") then fail "%s" text;
    emit st "%smov.pred %s, %s;" g (pd (reg (a 0))) (pu (reg (a 4)))
  | "ELECT" -> emit st "%select.sync _|%s, 0xffffffff;" g (pd (reg (a 0)))
  (* ---- special registers, constants ---- *)
  | "S2R" | "S2UR" ->
    let sr =
      match a 1 with
      | "SR_TID.X" -> "%tid.x"
      | "SR_CTAID.X" -> "%ctaid.x"
      | "SR_CgaCtaId" -> "%cluster_ctarank"
      | "SR_GLOBALTIMERLO" -> "%globaltimer_lo"
      | s -> fail "special register %s" s
    in
    emit st "%smov.u32 %s, %s;" g (dst (a 0)) sr
  | "LDC" | "LDC.64" | "LDCU.64" ->
    let off = int_of (bracket "c[0x0]" (a 1)) in
    let d = rnum (a 0) and f = (match reg (a 0) with U _ -> ud | _ -> rd) in
    if off = 0x37c then () (* the stack pointer: PTX's own *)
    else if off = 0x358 then (emit st "mov.b32 %s, 0;" (f d); emit st "mov.b32 %s, 0;" (f (d + 1))) (* the global descriptor: ptxas's own *)
    else if off >= 0x380 && (off - 0x380) mod 8 = 0 && op <> "LDC" then begin
      emit st "%sld.param.u64 %%d0, [p%d];" g ((off - 0x380) / 8);
      emit st "%smov.b64 {%s, %s}, %%d0;" g (f d) (f (d + 1))
    end
    else fail "constant 0x%x" off
  (* ---- memory ---- *)
  | "LDS" | "LDS.128" ->
    let base, off = addr (a 1) in
    let b = src base in
    if op = "LDS" then emit st "%sld.shared::cta.u32 %s, [%s+%d];" g (dst (a 0)) b off
    else let d = rnum (a 0) in emit st "%sld.shared::cta.v4.u32 {%s, %s, %s, %s}, [%s+%d];" g (rd d) (rd (d + 1)) (rd (d + 2)) (rd (d + 3)) b off
  | "STS" | "STS.128" ->
    let base, off = addr (a 0) in
    let b = src base in
    if op = "STS" then emit st "%sst.shared::cta.u32 [%s+%d], %s;" g b off (src (a 1))
    else let d = rnum (a 1) in emit st "%sst.shared::cta.v4.u32 [%s+%d], {%s, %s, %s, %s};" g b off (ru d) (ru (d + 1)) (ru (d + 2)) (ru (d + 3))
  | "STG.E" | "STG.E.128" | "STG.E.STRONG.GPU" | "LDG.E.STRONG.GPU" | "LDG.E.128.STRONG.GPU" ->
    let load = starts "LDG" op in
    let base, off = gaddr (if load then a 1 else a 0) in
    let p = pair ru base in
    let vec = contains ".128" op in
    let sem = if Filename.check_suffix op ".STRONG.GPU" then ".relaxed.gpu" else "" in
    let d = rnum (if load then a 0 else a 1) in
    let f = if load then rd else ru in
    let v = if vec then pf "{%s, %s, %s, %s}" (f d) (f (d + 1)) (f (d + 2)) (f (d + 3)) else f d in
    if load then emit st "%sld%s.global%s.u32 %s, [%s+%d];" g sem (if vec then ".v4" else "") v p off
    else emit st "%sst%s.global%s.u32 [%s+%d], %s;" g sem (if vec then ".v4" else "") p off v
  | "MEMBAR.ALL.GPU" -> emit st "%sfence.acq_rel.gpu;" g
  | "FENCE.VIEW.ASYNC.S" -> emit st "%sfence.proxy.async.shared::cta;" g
  | "FENCE.VIEW.ASYNC.G" -> emit st "%sfence.proxy.async.global;" g
  (* ---- barriers, the warp, control ---- *)
  | "BAR.SYNC.DEFER_BLOCKING" ->
    wait_ld st;
    emit st "tcgen05.fence::before_thread_sync;";
    (match args with
     | [ b ] -> emit st "%sbarrier.sync.aligned %s;" g (src b)
     | [ b; n ] -> emit st "%sbarrier.sync.aligned %s, %s;" g (src b) (src n)
     | _ -> fail "%s" text);
    emit st "tcgen05.fence::after_thread_sync;"
  | "WARPSYNC.ALL" -> emit st "%sbar.warp.sync -1;" g
  | "NANOSLEEP" -> emit st "%snanosleep.u32 %s;" g (src (a 0))
  (* dependent launch: the next grid in the stream may begin; this grid waits
     for the grids before it and their memory *)
  | "PREEXIT" -> emit st "%sgriddepcontrol.launch_dependents;" g
  | "ACQBULK" -> emit st "%sgriddepcontrol.wait;" g
  | "UCGABAR_ARV" ->
    wait_ld st;
    if st.inits then emit st "fence.mbarrier_init.release.cluster;";
    emit st "tcgen05.fence::before_thread_sync;";
    (* relaxed: the SASS's arrival orders no memory of its own; PTX's
       default, release, is MEMBAR.ALL.GPU and two error barriers before it *)
    emit st "%sbarrier.cluster.arrive.relaxed.aligned;" g
  | "UCGABAR_WAIT" -> emit st "%sbarrier.cluster.wait.aligned;" g; emit st "tcgen05.fence::after_thread_sync;"
  | "BRA" -> wait_ld st; emit st "%sbra %s;" g (a 0)
  | "EXIT" -> wait_ld st; emit st "%sexit;" g
  | "DEPBAR.LE" ->
    (* until at most n of this warp's bulk groups still read shared memory *)
    if a 0 <> "SB0" then fail "%s" text;
    emit st "%scp.async.bulk.wait_group.read %s;" g (src (a 1));
    emit st "bar.warp.sync -1;"
  (* ---- mbarriers ---- *)
  | "SYNCS.EXCH.64" ->
    let base, off = addr (a 1) in
    let w = rnum (a 2) in
    (* the count: 2^20 less the low word's bits 20:1; the high word's read
       too, as the SASS reads the pair *)
    emit st "shr.u32 %%t2, %s, 1;" (uu w);
    ignore (uu (w + 1));
    emit st "and.b32 %%t2, %%t2, 0xfffff;";
    emit st "sub.u32 %%t2, 1048576, %%t2;";
    emit st "%smbarrier.init.shared::cta.b64 [%s+%d], %%t2;" (one_lane ()) (src base) off;
    st.inits <- true
  | "SYNCS.PHASECHK.TRANS64.TRYWAIT" ->
    let base, off = addr (a 1) in
    let par = match a 2 with "RZ" -> "0" | p -> emit st "%sshr.u32 %%t3, %s, 31;" g (src p); "%t3" in
    (* the sleep while the phase is not complete: the suspend-time hint *)
    let hint = match List.nth_opt args 3 with Some n -> ", " ^ src n | None -> "" in
    emit st "%smbarrier.try_wait.parity.shared::cta.b64 %s, [%s+%d], %s%s;" g (pd (reg (a 0))) (src base) off par hint;
    emit st "tcgen05.fence::after_thread_sync;"
  | "SYNCS.ARRIVE.TRANS64" | "SYNCS.ARRIVE.TRANS64.RED" | "SYNCS.ARRIVE.TRANS64.A1T0" | "SYNCS.ARRIVE.TRANS64.RED.A1T0" ->
    wait_ld st;
    let base, off = addr (a 1) in
    let remote = starts "SYNCS.ARRIVE.TRANS64.RED" op and tx = not (Filename.check_suffix op ".A1T0") in
    emit st "tcgen05.fence::before_thread_sync;";
    let space = if remote then "shared::cluster" else "shared::cta" in
    if tx then emit st "%smbarrier.arrive.expect_tx.%s.b64 _, [%s+%d], %s;" g space (src base) off (src (a 2))
    else emit st "%smbarrier.arrive.%s.b64 _, [%s+%d];" g space (src base) off
  (* ---- tensor memory, the tensor core ---- *)
  | "UVIRTCOUNT.DEALLOC.SMPOOL" -> emit st "%stcgen05.relinquish_alloc_permit.cta_group::%d.sync.aligned;" g st.grp
  | "UTCATOMSWS.AND" ->
    (* the mask keeps what it does not free: the chunks freed are its low
       16 bits' zeros, the first of them the address *)
    if a 1 = "URZ" then (emit st "mov.b32 %%t4, 0;"; emit st "%stcgen05.dealloc.cta_group::%d.sync.aligned.b32 %%t4, 512;" g st.grp)
    else begin
      emit st "not.b32 %%t4, %s;" (src (a 1));
      emit st "and.b32 %%t4, %%t4, 0xffff;";
      emit st "popc.b32 %%t5, %%t4;";
      emit st "shl.b32 %%t5, %%t5, 5;";
      emit st "brev.b32 %%t6, %%t4;";
      emit st "clz.b32 %%t6, %%t6;";
      emit st "shl.b32 %%t6, %%t6, 5;";
      emit st "%stcgen05.dealloc.cta_group::%d.sync.aligned.b32 %%t6, %%t5;" g st.grp
    end
  | "UTCHMMA" | "UTCHMMA.2CTA" ->
    let ra = rnum (bracket "gdesc" (a 0)) and rb = rnum (bracket "gdesc" (a 1)) and rdd = rnum (bracket "tmem" (a 2)) and ri = rnum (bracket "idesc" (a 4)) in
    let da = pair uu ra and db = pair uu rb in
    ignore (uu (rnum (bracket "tmem" (a 3))));
    let en = match reg (a 5) with Q (false, 7) -> "%ptrue" | Q (true, 7) -> "%pfalse" | q -> if pneg q then fail "%s" text else pu q in
    let grp = if op = "UTCHMMA.2CTA" then 2 else 1 in
    emit st "%stcgen05.mma.cta_group::%d.kind::f16 [%s], %s, %s, %s, %s;" (one_lane ()) grp (uu rdd) da db (uu ri) en
  | "UTCBAR" -> emit st "%stcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%s];" (one_lane ()) (src (bracket "" (a 0)))
  | "UTCBAR.2CTA.MULTICAST" ->
    emit st "cvt.u16.u32 %%h0, %s;" (src (a 2));
    emit st "%stcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%s], %%h0;" (one_lane ()) (src (bracket "" (a 0)))
  | _ when starts "LDTM.x" op ->
    let n = num "LDTM.x" op in
    let base, off = addr (String.sub (a 1) 4 (String.length (a 1) - 4)) in
    let d = rnum (a 0) in
    emit st "add.u32 %%t7, %s, %d;" (src base) off;
    emit st "%stcgen05.ld.sync.aligned.32x32b.x%d.b32 {%s}, [%%t7];" g n (String.concat ", " (List.init n (fun k -> rd (d + k))));
    st.pending <- List.init n (fun k -> d + k) @ st.pending
  (* ---- the copy engine, the scheduler ---- *)
  | "UTMALDG.2D" | "UTMALDG.2D.2CTA" | "UTMALDG.2D.MULTICAST" | "UTMALDG.2D.MULTICAST.2CTA" ->
    let q = rnum (bracket "" (a 0)) and m = pair uu (rnum (bracket "" (a 1))) in
    let two = Filename.check_suffix op ".2CTA" and mc = contains "MULTICAST" op in
    if mc then (emit st "shr.u32 %%t0, %s, 16;" (src (a 2)); emit st "cvt.u16.u32 %%h1, %%t0;");
    emit st "%scp.async.bulk.tensor.2d%s.shared::cluster.global.mbarrier::complete_tx::bytes%s [%s], [%s, {%s, %s}], [%s]%s;" (one_lane ())
      (if two then ".cta_group::2" else "") (if mc then ".multicast::cluster" else "") (uu q) m (uu (q + 2)) (uu (q + 3)) (uu (q + 1))
      (if mc then ", %h1" else "")
  | "UTMASTG.2D" ->
    let q = rnum (bracket "" (a 0)) and m = pair uu (rnum (bracket "" (a 1))) in
    emit st "%scp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%s, {%s, %s}], [%s];" (one_lane ()) m (uu (q + 1)) (uu (q + 2)) (uu q)
  (* the same box added to what the matrix holds, in the L2 *)
  | "UTMAREDG.2D.ADD" ->
    let q = rnum (bracket "" (a 0)) and m = pair uu (rnum (bracket "" (a 1))) in
    emit st "%scp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%s, {%s, %s}], [%s];" (one_lane ()) m (uu (q + 1)) (uu (q + 2)) (uu q)
  | "UTMACMDFLUSH" -> emit st "%scp.async.bulk.commit_group;" g
  | "UTMACCTL.PF" -> let m = pair uu (rnum (bracket "" (a 0))) in emit st "%sprefetch.tensormap [%s];" (one_lane ()) m
  | "UGETNEXTWORKID.BROADCAST" ->
    emit st "%sclusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%s], [%s];" (one_lane ())
      (src (bracket "" (a 0))) (src (bracket "" (a 1)))
  | _ -> fail "no PTX for %s" text);
  match st.skip with Some l -> Buffer.add_string st.out (l ^ ":\n"); st.skip <- None | None -> ()

(* Tensor memory's allocation, as the program writes it: an elected lane
   asks until the allocator answers, the warp meets after. From the ELECT to
   the WARPSYNC it is one tcgen05.alloc, its answer read back into the
   register the construct leaves it in. *)
let alloc_region (items : line array) i =
  match items.(i) with
  | Ins { op = "ELECT"; _ } ->
    let n = Array.length items in
    let rec find j = if j >= n then None else match items.(j) with Ins { op = "WARPSYNC.ALL"; _ } -> Some j | _ -> find (j + 1) in
    (match find i with
     | Some e ->
       let body = Array.to_list (Array.sub items i (e - i + 1)) in
       let alloc = List.find_map (function Ins { op; _ } when starts "UTCATOMSWS.FIND_AND_SET" op || starts "UTCATOMSWS.2CTA.FIND_AND_SET" op -> Some op | _ -> None) body in
       let umov = List.find_map (function Ins { op = "UMOV"; args = [ d; v ]; _ } -> Some (num "UR" d, int_of v) | _ -> None) body in
       (match alloc, umov with Some op, Some (d, chunks) -> Some (e, op, d, chunks) | _ -> None)
     | None -> None)
  | _ -> None

(* ---- the kernel ---- *)

(* the instructions as units of the lift: a label, an instruction, or an
   allocation construct *)
type unit_ =
  | ULab of string
  | UIns of { guard : opnd option; op : string; args : string list; text : string }
  | UAlloc of { op : string; dreg : int; chunks : int }

let units items =
  let out = ref [] and i = ref 0 in
  while !i < Array.length items do
    (match alloc_region items !i with
     | Some (e, op, d, chunks) -> out := UAlloc { op; dreg = d; chunks } :: !out; i := e
     | None ->
       (match items.(!i) with
        | Lab l -> out := ULab l :: !out
        | Ins { guard; op = "SYNCS.PHASECHK.TRANS64.TRYWAIT" as op; args = p :: _ as args; text } ->
          (* a try-wait and its sleep while the phase is not complete: one
             PTX try-wait, the sleep its hint *)
          (match items.(!i + 1) with
           | Ins { guard = Some (P (true, n)); op = "NANOSLEEP.SYNCS"; args = [ t ]; _ } when reg p = P (false, n) ->
             out := UIns { guard; op; args = args @ [ t ]; text } :: !out; incr i
           | _ -> out := UIns { guard; op; args; text } :: !out)
        | Ins { guard; op; args; text } -> out := UIns { guard; op; args; text } :: !out));
    incr i
  done;
  Array.of_list (List.rev !out)

let emit_unit nm st scratch = function
  | ULab l -> wait_ld st; Buffer.add_string st.out (l ^ ":\n")
  | UIns { guard; op; args; text } -> lift_ins nm st guard op args text
  | UAlloc { op; dreg; chunks } ->
    (* the answer lands past the kernel's own shared memory *)
    ignore op;
    emit st "mov.u32 %%t0, lift_smem;";
    emit st "add.u32 %%t0, %%t0, %d;" scratch;
    emit st "tcgen05.alloc.cta_group::%d.sync.aligned.shared::cta.b32 [%%t0], %d;" st.grp (32 * chunks);
    emit st "bar.warp.sync -1;";
    emit st "ld.shared::cta.u32 %s, [%%t0];" (nm.def (KU dreg))

(* each unit's reads and writes, recorded off the lift itself *)
let effects scratch u =
  let uses = ref [] and defs = ref [] in
  let nm = { use = (fun k -> uses := k :: !uses; "%rec"); def = (fun k -> defs := k :: !defs; "%rec") } in
  let st = { out = Buffer.create 256; pending = []; inits = false; tmp = 0; skips = 0; skip = None; grp = 1 } in
  emit_unit nm st scratch u;
  List.sort_uniq compare !defs, List.sort_uniq compare !uses

(* The live ranges: over the control-flow graph, the definitions of each
   register that reach each read; the definitions one read sees are one
   value. A guarded write may not happen, so it kills nothing. *)
let webs (us : unit_ array) eff =
  let n = Array.length us in
  let labels = Hashtbl.create 64 in
  Array.iteri (fun i u -> match u with ULab l -> Hashtbl.replace labels l i | _ -> ()) us;
  let succ i =
    let next = if i + 1 < n then [ i + 1 ] else [] in
    match us.(i) with
    | UIns { op = "BRA"; args = [ l ]; guard; _ } ->
      let t = match Hashtbl.find_opt labels l with Some t -> t | None -> fail "no label %s" l in
      t :: (if guard = None then [] else next)
    | UIns { op = "EXIT"; guard = None; _ } -> []
    | _ -> next
  in
  (* the definitions: (unit, register) *)
  let parent = Hashtbl.create 1024 in
  let rec find x = match Hashtbl.find_opt parent x with Some p when p <> x -> let r = find p in Hashtbl.replace parent x r; r | _ -> x in
  let union a b = let ra = find a and rb = find b in if ra <> rb then Hashtbl.replace parent ra rb in
  let module M = Map.Make (struct type t = key let compare = compare end) in
  let module S = Set.Make (Int) in
  let din = Array.make n None in
  din.(0) <- Some M.empty;
  let reach = Array.make n M.empty in
  let work = Queue.create () in
  Queue.push 0 work;
  while not (Queue.is_empty work) do
    let i = Queue.pop work in
    let st = Option.get din.(i) in
    let defs, uses = eff.(i) in
    reach.(i) <- List.fold_left (fun m k -> M.add k (Option.value (M.find_opt k st) ~default:S.empty) m) M.empty uses;
    let guarded = match us.(i) with UIns { guard = Some _; _ } -> true | _ -> false in
    let st' = List.fold_left (fun m k -> M.add k (if guarded then S.add i (Option.value (M.find_opt k m) ~default:S.empty) else S.singleton i) m) st defs in
    List.iter
      (fun j ->
        match din.(j) with
        | None -> din.(j) <- Some st'; Queue.push j work
        | Some old ->
          let merged = M.union (fun _ a b -> Some (S.union a b)) old st' in
          if not (M.equal S.equal merged old) then (din.(j) <- Some merged; Queue.push j work))
      (succ i)
  done;
  (* a read's definitions are one value *)
  Array.iteri (fun i m -> ignore i; M.iter (fun k ds -> match S.elements ds with d :: rest -> List.iter (fun d' -> union (d', k) (d, k)) rest | [] -> ()) m) reach;
  let names = Hashtbl.create 1024 and count = ref 0 and pcount = ref 0 in
  let name_of (d, k) =
    let r = find (d, k) in
    match Hashtbl.find_opt names r with
    | Some s -> s
    | None ->
      let s = match k with KP _ | KQ _ -> incr pcount; pf "%%c%d" !pcount | _ -> incr count; pf "%%v%d" !count in
      Hashtbl.replace names r s;
      s
  in
  (* a read no definition reaches: a value of its own *)
  let undef = Hashtbl.create 8 in
  let use_name i k =
    match M.find_opt k reach.(i) with
    | Some ds when not (S.is_empty ds) -> name_of (S.min_elt ds, k)
    | _ -> (match Hashtbl.find_opt undef k with Some s -> s | None -> let s = name_of (-1 - Hashtbl.length undef, k) in Hashtbl.replace undef k s; s)
  in
  let namer i = { use = use_name i; def = (fun k -> name_of (i, k)) } in
  namer, (fun () -> !count, !pcount)

(* the CTAs an SM holds by shared memory -- 228 KB, each CTA's own and the
   1 KB the system reserves for it: the occupancy ptxas allocates registers
   for. Left to itself it assumes more CTAs than fit and spills to the stack *)
let ctas_per_sm h = max 1 (233472 / (h.smem + 16 + 1024))

(* The .cta_group the kernel's tcgen05 instructions share -- PTX allows one
   in a kernel: an allocation for the CTA pair, a pair MMA and a commit to
   the pair are 2, their one-CTA forms 1. A kernel whose instructions name
   both is refused. *)
let cta_group lines =
  let group l =
    match parse_line l with
    | Some (Ins { op; _ }) ->
      if List.mem op [ "UTCHMMA.2CTA"; "UTCBAR.2CTA.MULTICAST"; "UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN" ] then Some 2
      else if List.mem op [ "UTCHMMA"; "UTCBAR"; "UTCATOMSWS.FIND_AND_SET.ALIGN" ] then Some 1
      else None
    | _ -> None
  in
  match List.sort_uniq compare (List.filter_map group lines) with
  | [] -> 1
  | [ g ] -> g
  | _ -> fail "the kernel's tcgen05 instructions name both CTA groups; PTX allows one in a kernel"

let ptx (lines : string list) =
  let h = parse_header lines in
  let us = units (Array.of_list (List.filter_map parse_line lines)) in
  let scratch = h.smem in
  let eff = Array.map (effects scratch) us in
  let namer, counts = webs us eff in
  let st = { out = Buffer.create 65536; pending = []; inits = false; tmp = 0; skips = 0; skip = None; grp = cta_group lines } in
  Array.iteri (fun i u -> emit_unit (namer i) st scratch u) us;
  let nv, nc = counts () in
  let params = String.concat ",\n" (List.init h.params (fun k -> pf "\t.param .u64 p%d" k)) in
  String.concat "\n"
    [ "// lifted from warpc's SASS by warpc lift"
    ; pf "// launch: %d threads, cluster %d, dynamic shared memory %d bytes" h.threads h.cluster (h.smem + 16)
    (* sm_103a is a target from PTX ISA 8.8 on *)
    ; (if h.sm = "sm_100a" then ".version 8.7" else ".version 8.8"); ".target " ^ h.sm; ".address_size 64"; ""
    ; ".extern .shared .align 1024 .b8 lift_smem[];"; ""
    ; pf ".visible .entry %s(\n%s\n)" h.name params
    ; pf ".reqntid %d, 1, 1" h.threads
    ; pf ".minnctapersm %d" (ctas_per_sm h)
    ; (if h.cluster > 1 then pf ".reqnctapercluster %d, 1, 1" h.cluster else "")
    ; "{"
    ; pf "\t.reg .b32 %%v<%d>;" (nv + 1); pf "\t.reg .pred %%c<%d>;" (nc + 1); "\t.reg .b64 %d<8>;"
    ; "\t.reg .b32 %t<8>;"; "\t.reg .b16 %h<2>;"; "\t.reg .pred %ptrue, %pfalse, %pe, %pg, %pw;"
    ; "\tsetp.eq.u32 %ptrue, 0, 0;"; "\tsetp.ne.u32 %pfalse, 0, 0;"
    ; "\t// the SASS addresses shared memory from 0x400: the window must start there;"
    ; "\t// in a cluster the address carries the CTA's rank from bit 24"
    ; "\tmov.u32 %t0, lift_smem;"; "\tand.b32 %t0, %t0, 0xffffff;"; "\tsetp.ne.u32 %pw, %t0, 1024;"; "\t@%pw trap;"
    ; "\t// each warp's one lane for the work it issues once"; "\telect.sync _|%pe, 0xffffffff;"
    ; Buffer.contents st.out; "}"; "" ]
