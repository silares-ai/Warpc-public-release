(** SASS instructions as data. An instruction is its text as cuobjdump prints
   it -- literal pieces and register operands -- and what the compiler needs
   to know about it: which registers it writes, reads, and reads late (after
   issue), its guard, its effect on the registers for the host-side
   interpreter, and the control-flow it takes. *)

type reg = R of int | UR of int | P of int | UP of int

type branch = [ `Bra of string | `Exit | `No ]

type piece = Text of string | Reg of reg | Lab of string

type origin = Program | Derived | Construct

type machine = {
  get : reg -> int;
  set : reg -> int -> unit;
  special : string -> int;
  const : int -> int;
  load : int -> int;
  store : int -> int -> unit;
  gload : int -> int;
  gstore : int -> int -> unit;
  alloc : int -> int;
  lane : int;
}

(** What an instruction does outside the registers that orders the warps --
    the barriers' phases, the copy engine, the tensor core, the proxies --
    with its operands' values, for the interpreter's run of a cluster's
    warps together ({!Interp}). Addresses are shared::cluster addresses, the
    CTA in the top byte; a tensor-memory address is lane << 16 | column. *)
type act =
  | Init of { bar : int; count : int }  (** a barrier, the arrivals its phases take *)
  | Arrive of { bar : int; tx : int }  (** a lane's arrival, stating [tx] bytes the phase also waits for *)
  | Wait of { bar : int; parity : int; ahead : bool }
      (** until the phase of [parity] has completed; [ahead], asked now and read later *)
  | Load of { dst : int; bar : int; map : int; mask : int option; two : bool }  (** the copy engine's load of [map]'s box *)
  | Store of { src : int; map : int; x : int; y : int; add : bool }
      (** its store of a box from shared memory, to [map]'s matrix at column [x], row [y];
          [add], added to what is there, in the L2 *)
  | Mma of { accumulate : bool }
      (** the tensor core's MMA, adding to the accumulator or overwriting it; what it reads and writes is the kernel's touches *)
  | Global of { addr : int; bytes : int; write : bool; strong : bool }
      (** a lane's access to global memory; [strong], at the GPU's coherence point *)
  | Gpu_fence  (** this thread's writes so far, visible to the whole GPU before any after *)
  | Commit of { bar : int; mask : int option }
      (** an arrival once the MMAs before it have run: on [bar], or at its offset in each CTA of [mask] *)
  | Flush  (** the copy engine's stores so far, a group *)
  | Drain of int  (** until at most this many groups are still reading shared memory *)
  | Proxy_fence  (** this thread's writes so far, visible to the copy engine and the tensor core *)
  | Cluster_arrive
  | Cluster_wait
  | Request of { answer : int; bar : int }
      (** the scheduler's answer, 16 bytes at a window offset in every CTA, reported on [bar] there *)
  | Tmem_load of { addr : int; n : int }  (** [n] columns of the warp's lane quarter *)
  | Tmem_free of int  (** the 32-column chunks freed, a mask *)

type insn = {
  id : int;
  asm : piece list;
  guard : (reg * bool) option;
  defs : reg list;
  uses : reg list;
  late_uses : reg list;
  branch : branch;
  named_barrier : int option;
  sem : (machine -> unit) option;
  act : (machine -> act) option;
  origin : origin;
}

type item = Label of string | I of insn

type t = { mutable items : item list; }

val create : unit -> t

(** An instruction's text as pieces: its register operands and the literal
    text between. *)
val pieces : string -> piece list

(** An instruction's name: its first word, the guard apart. *)
val opcode : insn -> string

(** The operand of an instruction's text that reads the register, by its
    place among the register operands, and the register's place in it: the
    last naming it (0), or else the last whose group of registers -- of
    [regs], the ones the instruction touches -- takes it in; -1 if none
    does. *)
val slot : ?regs:reg list -> piece list -> reg -> int * int

val items : t -> item list

val reg_text : reg -> string

(** the instruction as cuobjdump prints it, each register as [name] spells it *)
val to_string : ?name:(reg -> string) -> insn -> string

(** constant bank / special registers *)
val ldc : t -> int -> int -> unit

val s2r_tid : t -> int -> unit

val s2r_timer : t -> int -> unit

(** The SM's cycle counter into a pair. *)
val cs2r_clock : t -> int -> unit

val ldc64 : t -> int -> int -> unit

val ldcu64 : t -> int -> int -> unit

(** d:d+1 = a * imm + c:c+1, unsigned *)
val imad_wide : t -> int -> int -> int -> int -> unit

val stg32 : ?desc:int -> t -> base:int -> imm:int -> data:int -> unit

(** d = a * imm + c *)
val imad : t -> int -> int -> int -> int -> unit

val imad_rz : t -> int -> int -> int -> unit

val mov_rz : t -> int -> unit

val lop3_and : t -> int -> int -> int -> unit

val shf_r : t -> int -> int -> int -> unit

val bra : t -> ?neg:bool -> int -> string -> unit

val exit : t -> unit

(** ---- the tcgen05 path: uniform datapath, shared memory, mbarriers, TMEM ---- *)
val s2ur_cta : t -> int -> unit

val umov : t -> int -> int -> unit

val uiadd3 : t -> int -> int -> int -> unit

val uiadd3_uu : t -> int -> int -> int -> unit

val ushf_l : t -> int -> int -> int -> unit

val ushf_r : t -> int -> int -> int -> unit

val ulop3_and : t -> int -> int -> int -> unit

val ulop3_not : t -> int -> int -> unit

(** d = (a << sh) + c *)
val ulea : t -> int -> int -> int -> int -> unit

(** d = a * imm + c, as ptxas spells it (UIMAD UR5, UR5, -0xa0, URZ) *)
val uimad_imm : t -> int -> int -> int -> int -> unit

(** d = a * imm, uniform: UIMAD UR, UR, imm, URZ *)
val uimad_z : t -> int -> int -> int -> unit

val r2ur : t -> int -> int -> unit

val mov_ur : t -> int -> int -> unit

(** d = (a << sh) + c *)
val lea_ur : t -> int -> int -> int -> int -> unit

val mov_imm : t -> int -> int -> unit

val lop3_xor : t -> int -> int -> int -> unit

val iadd3 : t -> int -> int -> int -> unit

val iadd3_c : t -> int -> int -> int -> unit

val isetp_ne_u32 : t -> int -> int -> int -> unit

val jmp : t -> string -> unit

(** shared memory; addresses are the cluster form (cta << 24 | offset) held in a UR *)
val lds : t -> int -> ur:int -> imm:int -> unit

val sts : t -> width:int -> r:int -> ur:int -> imm:int -> data:int -> unit

val sts_ur : t -> ur:int -> imm:int -> data:int -> unit

val stg128 : ?desc:int -> t -> base:int -> imm:int -> data:int -> unit

val fence_view_async : t -> unit

(** FENCE.VIEW.ASYNC.G: global memory the generic proxy wrote, visible to
    the copy engine's accesses after it, and the other way round *)
val fence_view_async_global : t -> unit

(** a named barrier over a subset of the block: the warps that store share one,
   and the warps that feed the tensor core never arrive at it *)
val bar_sync_n : t -> bar:int -> count:int -> unit

val syncs_exch : t -> base:int -> imm:int -> v:int -> unit

(** a test of a barrier's phase, the answer in a predicate *)
val syncs_trywait :
  t -> int -> base:int -> imm:int -> parity_reg:int option -> unit

(** A try-wait whose answer is read later, a stage ahead: the barrier unit
   reads its operands until it answers, so they are read late. *)
val syncs_trywait_ahead :
  t -> int -> base:int -> imm:int -> parity:int -> unit

val utcatomsws_fas : ?two:bool -> ?up:int -> t -> int -> unit

val plop3_up : t -> int -> up:int -> unit

val nanosleep : t -> unit

(** a failed barrier test sleeps until the barrier unit has news,
   instead of asking again at once *)
val nanosleep_syncs : ?ns:int -> t -> neg:bool -> int -> unit

val uvirtcount_dealloc : t -> unit

val utcatomsws_and : t -> int -> unit

(** clear the whole allocation bitmap: all 512 columns freed at once *)
val utcatomsws_clear : t -> unit

val utcbar : t -> mbar:int -> unit

val ushf_l_ur : t -> int -> int -> int -> unit

val ulop3_or_ur : t -> int -> int -> int -> unit

(** ---- the pipelined path: TMA, per-stage mbarriers, uniform predicates ---- *)
val s2r_ctaid : t -> int -> axis:string -> unit

val isetp_eq_u32 : t -> int -> int -> int -> unit

val isetp_lt_u32_imm : t -> int -> int -> int -> unit

val lop3_xor_imm : t -> int -> int -> int -> unit

(** [two] marks a load that feeds a two-CTA MMA: its bytes are reported to the
   pair, so neither CTA can start the MMA until both halves have landed *)
val utmaldg : ?two:bool -> t -> g:int -> map:int -> unit

val utmaldg_mc :
  ?guard:int -> ?two:bool -> t -> g:int -> map:int -> mask:int -> unit

val syncs_arrive_tx_red :
  t -> guard:int -> base:int -> imm:int -> tx:int -> unit

val syncs_arrive_tx : t -> guard:int -> base:int -> imm:int -> tx:int -> unit

val syncs_arrive_red : t -> guard:int -> base:int -> imm:int -> unit

val syncs_arrive : t -> guard:int -> base:int -> imm:int -> unit

(** the same with the address in a register, [Rn+URZ], as ptxas addresses a
   barrier it indexes at run time (ref/clcloop.ptx) *)
val syncs_arrive_r : t -> guard:int -> addr:int -> unit

val syncs_arrive_tx_r : t -> guard:int -> addr:int -> tx:int -> unit

val syncs_arrive_red_r : t -> guard:int -> addr:int -> unit

val syncs_arrive_tx_red_r : t -> guard:int -> addr:int -> tx:int -> unit

val syncs_trywait_r : t -> int -> addr:int -> parity:int -> unit

(** A CTA of a cluster may not leave while a peer can still write its shared
   memory, so the cluster meets once before the exits.  This is what
   barrier.cluster.arrive / barrier.cluster.wait lower to (ref/remarrive.ptx).
   The meeting split in two, as ptxas places it: the arrival once
   the barriers are initialised -- it publishes them to the cluster -- and the
   wait just before the first use of a barrier, with the work between. *)
val cluster_arrive_inits : t -> unit

val cluster_wait : t -> unit

(** without a cluster, the CTA's barrier publishes the initialisations *)
val bar_sync_inits : t -> unit

(** cluster launch control's request: cancel a cluster not yet launched,
   its first CTA's id written to [resp] in every CTA of this cluster *)
val ugetnextworkid : t -> resp:int -> mbar:int -> unit

val lds128_r : t -> int -> addr:int -> unit

val uisetp_ne : t -> int -> int -> unit

val uisetp_ne_imm : t -> int -> int -> int -> unit

val uisetp_eq0 : t -> int -> int -> unit

val utchmma_up :
  t -> a:int -> bb:int -> d:int -> e:int -> idesc:int -> up:int -> unit

val utchmma_acc :
  t -> a:int -> bb:int -> d:int -> e:int -> idesc:int -> acc:bool -> unit

val utchmma2 :
  t ->
  guard:int ->
  a:int -> bb:int -> d:int -> e:int -> idesc:int -> acc:bool -> unit

val utchmma2_up :
  t ->
  guard:int ->
  a:int -> bb:int -> d:int -> e:int -> idesc:int -> up:int -> unit

val utcbar2_mc : t -> guard:int -> mbar:int -> mask:int -> unit

val ldtm_off : t -> int -> n:int -> addr:int -> imm:int -> unit

val utmacctl_pf : t -> map:int -> unit

(** the lowest active lane, for the code that must run once per warp *)
val elect : t -> int -> unit

(** shared-memory staging for the epilogue: a warp writes its rows, then reads
   them back so that consecutive lanes hold consecutive addresses *)
val sts_r : t -> width:int -> r:int -> imm:int -> data:int -> unit

val warpsync : t -> unit

(** Dependent launch: PREEXIT lets the next grid in the stream begin;
    ACQBULK waits for every grid before this one to finish, its memory
    visible. A kernel touches no global memory before its ACQBULK. *)
val preexit : t -> unit
val acqbulk : t -> unit

val mov_rr : t -> int -> int -> unit

(** UTMASTG.2D [URg], [URmap]: URg = shared source, URg+1 = x, URg+2 = y --
   three of the aligned quad. The copy engine writes global memory, so a warp
   issues one instruction for its whole block instead of a store per row.
   [add]: UTMAREDG.2D.ADD, the block added to what the matrix holds, in the
   L2. *)
val utmastg : ?add:bool -> t -> g:int -> map:int -> unit

(** the copy engine's stores so far, closed as a group (cp.async.bulk.commit_group) *)
val utmacmdflush : t -> unit

(** wait for the copy engine to have taken the staged block before it is reused *)
val depbar_drain : t -> unit

(** wait until at most [n] groups of tensor-map stores are still reading
   shared memory (cp.async.bulk.wait_group.read): the groups finish in order,
   so this frees the staging copy the group [n + 1] back was reading *)
val depbar_le : t -> n:int -> unit

(** ---- for work split over CTAs: register compares, the partials' adds, and
   the flags one CTA hands another through global memory ---- *)
val isetp_rr : t -> cmp:string -> int -> int -> int -> unit

val iadd3_sub : t -> int -> int -> int -> unit

val fadd : t -> int -> int -> int -> unit

(** a load and a store at the GPU's coherence point, past the L1: what
   ld.relaxed.gpu / st.relaxed.gpu are *)
val ldg_strong : ?desc:int -> t -> int -> base:int -> imm:int -> unit

val stg_strong : ?desc:int -> t -> base:int -> imm:int -> data:int -> unit

(** every memory access before it is visible to the whole GPU before any after
   it; the L1's lines are dropped, so what another multiprocessor wrote is
   read from L2 -- the fence and the invalidate ptxas puts around a release
   and an acquire at GPU scope *)
val membar_gpu : t -> unit

val ldg_strong128 : ?desc:int -> t -> int -> base:int -> imm:int -> unit
