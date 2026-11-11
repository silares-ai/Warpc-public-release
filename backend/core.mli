(** The core language: a program is a list of statements, and a statement is
    one SASS instruction or a label -- nothing else. Its operands are
    registers the program declares by name, and a register's class and
    width are its type: every instruction states the registers it takes, so
    a uniform register where a general one belongs, or one register where a
    pair belongs, does not type-check. The compiler colours the registers
    ({!Color}) and stops with an error when they do not fit, and writes the
    header from what the program declares.
    It decides nothing else: the loops, the stages, the protocols between
    warps and the order of the instructions are the program's text. An
    address is not: a memory operand is a buffer at coordinates ({!Mem}),
    and the instructions that form its address are derived from the
    buffer's layout into the program's holes ({!Prog}).

    The instructions are {!Sass}'s constructors, under the same names. *)

(** {1 Registers} *)

type cls =
  | R
  | UR
  | P
  | UP

(** A declaration, untyped: what the allocator colours. *)
type raw = private
  { id : int (** a placeholder number until coloured; a wide register takes consecutive ones *)
  ; name : string
  ; cls : cls
  ; width : int
  ; align : int
  }

(** The class and width of a register, as its type. *)
type r = [ `R ]

type r2 = [ `R2 ]
type r4 = [ `R4 ]

(** A vector of general registers, its length a value, checked where it is
    used. *)
type rv = [ `Rv ]

type ur = [ `UR ]
type ur2 = [ `UR2 ]
type ur4 = [ `UR4 ]
type p = [ `P ]
type up = [ `UP ]

(** A declared register of class and width ['k]. *)
type +'k reg

val raw : 'k reg -> raw
val id : 'k reg -> int
val name : 'k reg -> string
val cls_of : 'k reg -> cls
val width : 'k reg -> int
val cls_name : cls -> string

(** The file a derived value is computed in. *)
type _ vclass =
  | General : r vclass
  | Uniform : ur vclass

val cls_of_vclass : 'k vclass -> cls

(** A fresh register of the file. *)
val fresh_in : 'k vclass -> string -> 'k reg

(** The declarations, each named; a pair or quad aligned. *)
val r : string -> r reg

val r2 : string -> r2 reg
val r4 : string -> r4 reg

(** [n] general registers, aligned to [align] or their size up to 8. *)
val rn : ?align:int -> int -> string -> rv reg

val ur : string -> ur reg
val ur2 : string -> ur2 reg
val ur4 : string -> ur4 reg
val p : string -> p reg
val up : string -> up reg

(** The [i]-th register of a wide general or uniform one. *)
val part : [< `R2 | `R4 | `Rv ] reg -> int -> r reg

val upart : [< `UR2 | `UR4 ] reg -> int -> ur reg

(** [w] consecutive registers of a vector from its [i]-th, aligned. *)
val slice : rv reg -> int -> int -> rv reg

(** Placeholder numbers start here; [owner] finds the declaration a
    placeholder unit belongs to while a program is being built. *)
val first_placeholder : int

val owner : int -> raw option

(** {1 Statements} *)

type stmt = Sass.item

(** [s] under a predicate, general or uniform. *)
val on : ?neg:bool -> [< `P | `UP ] reg -> stmt -> stmt

val label : string -> stmt

(** A label no other has. *)
val fresh_label : string -> string

(** The one instruction a Sass constructor pushes. *)
val one : (Sass.t -> unit) -> stmt

(** {1 The instructions} *)

(** The stack pointer, R1, the one register a program does not declare. *)
val stack_pointer : unit -> stmt

(** Where parameter [i], and the global-memory descriptor, sit in the
    constant bank. *)
val param_off : int -> int

val gmem_desc_off : int
val ldc64 : r2 reg -> int -> stmt
val ldcu64 : ur2 reg -> int -> stmt
val s2r_tid : r reg -> stmt
val s2r_timer : r reg -> stmt

(** The SM's cycle counter. *)
val cs2r_clock : r2 reg -> stmt
val s2r_ctaid : r reg -> axis:string -> stmt
val s2ur_cta : ur reg -> stmt

(** The general ALU. *)
val imad_wide : r2 reg -> r reg -> int -> r2 reg -> stmt

val imad : r reg -> r reg -> int -> r reg -> stmt
val imad_rz : r reg -> r reg -> int -> stmt
val mov_rz : r reg -> stmt
val mov_imm : r reg -> int -> stmt
val mov_rr : r reg -> r reg -> stmt
val mov_ur : r reg -> ur reg -> stmt
val lop3_and : r reg -> r reg -> int -> stmt
val lop3_xor_imm : r reg -> r reg -> int -> stmt
val shf_r : r reg -> r reg -> int -> stmt
val lea_ur : r reg -> r reg -> ur reg -> int -> stmt
val iadd3 : r reg -> r reg -> r reg -> stmt
val iadd3_c : r reg -> r reg -> int -> stmt
val iadd3_sub : r reg -> r reg -> r reg -> stmt
val fadd : r reg -> r reg -> r reg -> stmt
val isetp_lt_u32 : p reg -> r reg -> int -> stmt
val isetp_ne_u32 : p reg -> r reg -> int -> stmt
val isetp_eq_u32 : p reg -> r reg -> int -> stmt
val isetp_rr : cmp:string -> p reg -> r reg -> r reg -> stmt
val elect : p reg -> stmt
val plop3_up : p reg -> up reg -> stmt
val r2ur : ur reg -> r reg -> stmt

(** The uniform ALU. *)
val umov : ur reg -> int -> stmt

val uiadd3 : ur reg -> ur reg -> int -> stmt
val uiadd3_uu : ur reg -> ur reg -> ur reg -> stmt
val uimad_imm : ur reg -> ur reg -> int -> ur reg -> stmt
val uimad_z : ur reg -> ur reg -> int -> stmt
val ushf_l : ur reg -> ur reg -> int -> stmt
val ushf_r : ur reg -> ur reg -> int -> stmt
val ushf_l_ur : ur reg -> ur reg -> ur reg -> stmt
val ulop3_and : ur reg -> ur reg -> int -> stmt
val ulop3_or_ur : ur reg -> ur reg -> ur reg -> stmt
val ulop3_not : ur reg -> ur reg -> stmt
val ulea : ur reg -> ur reg -> ur reg -> int -> stmt
val uisetp_ne : up reg -> ur reg -> stmt
val uisetp_eq0 : up reg -> ur reg -> stmt
val uisetp_ne_imm : up reg -> ur reg -> int -> stmt

(** Control. *)
val bra : ?neg:bool -> p reg -> string -> stmt

val jmp : string -> stmt
val exit : unit -> stmt
val bar_sync_n : bar:int -> count:int -> stmt
val bar_sync_inits : unit -> stmt
val cluster_arrive_inits : unit -> stmt
val cluster_wait : unit -> stmt
val warpsync : unit -> stmt

(** Dependent launch: the next grid may begin (PREEXIT); every grid before
    has finished and its memory is visible (ACQBULK). *)
val preexit : unit -> stmt
val acqbulk : unit -> stmt
val fence_view_async : unit -> stmt
val fence_view_async_global : unit -> stmt
val membar_gpu : unit -> stmt
val depbar_drain : unit -> stmt
val depbar_le : n:int -> stmt
val nanosleep : unit -> stmt

(** Shared memory: a uniform base and an immediate, or a general address. *)
val lds : r reg -> base:ur reg -> imm:int -> stmt

val lds128_r : r4 reg -> addr:r reg -> stmt
val sts : width:int -> r:r reg -> base:ur reg -> imm:int -> data:rv reg -> stmt
val sts_ur : base:ur reg -> imm:int -> data:r reg -> stmt
val sts_r : width:int -> base:r reg -> imm:int -> data:rv reg -> stmt

(** Global memory: the descriptor a uniform pair, the address a general
    pair. *)
val stg32 : desc:ur2 reg -> base:r2 reg -> imm:int -> data:r reg -> stmt

val stg128 : desc:ur2 reg -> base:r2 reg -> imm:int -> data:rv reg -> stmt
val ldg_strong : desc:ur2 reg -> r reg -> base:r2 reg -> imm:int -> stmt
val ldg_strong128 : desc:ur2 reg -> rv reg -> base:r2 reg -> imm:int -> stmt
val stg_strong : desc:ur2 reg -> base:r2 reg -> imm:int -> data:r reg -> stmt

(** An mbarrier: its shared::cluster address, a uniform register and an
    immediate, or a general register. *)
type bar =
  | Ubar of ur reg * int
  | Rbar of r reg

val syncs_exch : bar -> v:ur2 reg -> stmt
val trywait : p reg -> bar -> parity:r reg -> stmt

(** Ask now, read the answer later ({!Sass.syncs_trywait_ahead}). *)
val trywait_ahead : p reg -> bar -> parity:r reg -> stmt

(** The arrivals: guards are put on by {!on}; [_red] on another CTA's copy. *)
val arrive_tx : bar -> tx:r reg -> stmt

val arrive_red : bar -> stmt
val arrive_tx_red : bar -> tx:r reg -> stmt
val sleep_syncs : ?ns:int -> unit -> stmt

(** Tensor memory and the tensor core. The second tensor-memory operand and
    the instruction descriptor are one field of the encoding, an aligned
    pair: [ei] = (e, idesc). *)
val utcatomsws_fas : ?two:bool -> up:up reg -> ur reg -> stmt

val utcatomsws_and : ur reg -> stmt
val utcatomsws_clear : unit -> stmt
val uvirtcount_dealloc : unit -> stmt
val utchmma : a:ur2 reg -> bb:ur2 reg -> d:ur reg -> ei:ur2 reg -> acc:bool -> stmt
val utchmma_up : a:ur2 reg -> bb:ur2 reg -> d:ur reg -> ei:ur2 reg -> up:up reg -> stmt
val utchmma2 : leader:up reg -> a:ur2 reg -> bb:ur2 reg -> d:ur reg -> ei:ur2 reg -> acc:bool -> stmt
val utchmma2_up : leader:up reg -> a:ur2 reg -> bb:ur2 reg -> d:ur reg -> ei:ur2 reg -> up:up reg -> stmt
val utcbar : bar:ur reg -> stmt
val utcbar2_mc : leader:up reg -> bar:ur reg -> mask:ur reg -> stmt
val ldtm : rv reg -> n:int -> addr:ur reg -> imm:int -> stmt

(** The copy engine: an operand quad and a tensor map. *)
val utmaldg : ?two:bool -> g:ur4 reg -> map:ur2 reg -> unit -> stmt

val utmaldg_mc : ?two:bool -> g:ur4 reg -> map:ur2 reg -> mask:ur reg -> unit -> stmt
val utmastg : g:ur4 reg -> map:ur2 reg -> stmt
val utmaredg : g:ur4 reg -> map:ur2 reg -> stmt
val utmacmdflush : unit -> stmt
val utmacctl_pf : map:ur2 reg -> stmt

(** A request to the scheduler: the answer's address and its barrier's, an
    aligned pair. *)
val ugetnextworkid : resp_bar:ur2 reg -> stmt

(** {1 The kernel} *)

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

(** What a derived operand must hold where its instruction runs: [operand]
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

(** What an instruction touches that its registers alone do not say -- an
    MMA's operand tiles and its accumulator -- read off the layouts it came
    from: runs of bytes of shared memory, at shared::cluster addresses, or
    runs of tensor-memory columns from a lane-quarter's first lane, at
    CTA << 24 | lane << 16 | column, where it runs. *)
type space = Smem | Tmem

type touch =
  { space : space
  ; write : bool
  ; runs : env -> (int * int) list
  }

(** A named block of the static window or the dynamic region: its first
    byte in a CTA's shared window, how many copies, the bytes between them. *)
type region =
  { rname : string
  ; first : int
  ; copies : int
  ; stride : int
  }

(** The declarations a body names. *)
val regs_of : stmt list -> raw list

(** A kernel: its launch, read off the program -- the threads its warps
    take, the barriers and shared memory it declares, whether it uses the
    tensor core -- its parameters, its body, the declarations the body
    names, and each derived operand's check, keyed by instruction. *)
type kernel =
  { kname : string
  ; params : param list
  ; threads : int
  ; grid : int * int
  ; cluster : int
  ; together : int (* consecutive clusters that wait on one another's global writes *)
  ; smem : int
  ; dynsmem : int
  ; mbarriers : int
  ; named_barriers : int
  ; tile : int * int * int
  ; tcgen05 : bool
  ; body : stmt list
  ; regs : raw list
  ; checks : (int * check) list
  ; touches : (int * touch) list
  ; regions : region list
  ; notes : string list
  ; timing : timing
  }

(** A timing build: the events the program stamps, or a stamp each time a
    stage lands ({!Stamp}). *)
and timing =
  | No_timing
  | Stamps of (int * string) list
  | Stage_stamps
