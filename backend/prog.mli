(** A program under construction: statements appended one call at a time,
    so a program reads as a sequence of lines, each one instruction.

    Two kinds of instruction are not the program's lines, and each is marked
    where it lands ({!Sass.origin}). The control of the language's
    constructs -- a loop's count, compare and branch; a ring's step; a
    role's dispatch; a spin's sleep and branch -- is emitted by the
    construct. The address arithmetic a layout implies goes into holes,
    filled when the program is finished, so a derived instruction lands
    where it belongs whatever order it was asked for in:
    - a value of indices that stay fixed is computed once, where the last of
      them (or the base register it adds to) is set, and shared from there;
    - a value affine in a counter is set where the loop starts and stepped
      where it steps;
    - a value affine in a ring position is one multiply-add from the stage,
      at the top of the step, ahead of the wait that follows.

    An index's kind -- fixed, a counter, a ring position -- and the file of
    its register are its type, so only a ring is stepped. *)

(** A program being built. *)
type t

val create : unit -> t

(** {1 Lines} *)

(** The program's instruction, under the guard {!when_} has set. *)
val emit : t -> Core.stmt -> unit

val emits : t -> Core.stmt list -> unit

(** The program's instruction whose derived operands must hold what the
    checks say where it runs ({!Interp}). *)
val emit_checked : t -> Core.stmt -> Core.check list -> unit

(** The checks of an instruction emitted another way. *)
val check : t -> Core.stmt -> Core.check list -> unit

(** What an instruction touches that its registers do not say ({!Core.touch}). *)
val touch : t -> Core.stmt -> Core.touch list -> unit

(** An instruction of a construct: the language's, not the program's line. *)
val construct : t -> Core.stmt -> unit

val constructs : t -> Core.stmt list -> unit

(** Instructions the compiler derives, here: an operand's address formed
    where the instruction takes it. *)
val derived : t -> Core.stmt list -> unit

val label : t -> string -> unit

(** A label no other has. *)
val fresh : string -> string

(** Every instruction [f] emits, under the predicate. *)
val when_ : t -> ?neg:bool -> [< `P | `UP ] Core.reg -> (unit -> unit) -> unit

(** {1 Indices} *)

type fixed = [ `Fixed ]
type counter = [ `Counter ]
type ring = [ `Ring ]

(** A run-time index of kind ['k] -- {!fixed}, {!counter} or {!ring} -- in a
    register of ['c]: a register with a range. *)
type ('k, 'c) index

(** An index of any kind, as a value's variable. *)
type var = Var : ('k, 'c) index -> var

val index_name : ('k, 'c) index -> string
val index_range : ('k, 'c) index -> int
val reg : ('k, 'c) index -> 'c Core.reg

(** An index the program has just set, fixed from here to the end of the
    scope it was set in. *)
val fixed : t -> name:string -> range:int -> cls:'c Core.vclass -> 'c Core.reg -> (fixed, 'c) index

(** This thread's warp and lane, from its index [tid] in the CTA; the
    program's warps are [warps]. *)
val warp_lane : t -> tid:Core.r Core.reg -> warps:int -> (fixed, Core.r) index * (fixed, Core.r) index

(** An index's value where a check runs: a counter's register is one
    ahead inside its loop. *)
val index_value : Core.env -> ('k, 'c) index -> int

val var_value : Core.env -> var -> int
val var_name : var -> string
val var_range : var -> int

(** The parity waits on a ring's barriers wait at; none for another kind. *)
val ring_parity : var -> Core.r Core.reg option

(** {1 Constructs} *)

(** A role: the warps that run [f], its constants computed at its start.
    With [skip], a compare into the predicate it is given: the warps it
    holds for branch past the role. *)
val role : t -> ?name:string -> ?skip:(Core.p Core.reg -> Core.stmt) -> (unit -> 'a) -> 'a

(** A spin: [test] -- the instruction that sets the predicate it is given
    when what it waits for has happened -- then, until it has, a sleep and
    a branch back. *)
val spin : t -> ?ns:int -> ?checks:Core.check list -> (Core.p Core.reg -> Core.stmt) -> unit

type count =
  | Times of int
  | Reg_times of Core.r Core.reg

(** For i = 0 .. count - 1, at least once. The count and the compare come
    first in the body, so the compare's latency runs under it. *)
val for_ : t -> name:string -> ?pred:Core.p Core.reg -> count:count -> ((counter, Core.r) index -> unit) -> unit

(** A loop over the values [reg] takes: the index is fixed for an
    iteration, its values computed at the top of each; [next] sets the next
    value and branches back to the label it is given. *)
val over : t -> name:string -> reg:Core.r Core.reg -> range:int -> next:(string -> 'a) -> ((fixed, Core.r) index -> unit) -> 'a

(** A persistent loop: from what [reg] holds, [step] at a time while below
    [until], the first always taken. *)
val stride : t -> name:string -> ?pred:Core.p Core.reg -> reg:Core.r Core.reg -> until:int -> step:int -> ((fixed, Core.r) index -> unit) -> unit

(** A ring position: the stage, a uniform register from 0, and the parity
    its barriers are waited at, starting at [parity]. *)
val ring : t -> name:string -> depth:int -> parity:int -> (ring, Core.ur) index

(** One step of a ring: [body] runs at the current stage -- its values of
    the ring computed at the top, whether this step wraps asked there too --
    and the ring moves to the next stage after it, flipping its parity when
    it wraps. A step inside a step of the same ring is refused. *)
val step : t -> (ring, Core.ur) index -> (unit -> 'a) -> 'a

(** {1 Derived values} *)

(** A register of either file with its file, as a value's base. *)
type some = Some_reg : 'c Core.vclass * 'c Core.reg -> some

(** The value of [e] + [base] in a register of file [cls], [e] over the
    indices [vars] names. A value of fixed indices is computed once, where
    the last of them is set -- or, of none, where the role starts -- and
    shared from there. A value affine in a counter follows it: set where it
    starts, stepped where it steps; one of a ring, one multiply-add from the
    stage at the top of the step, in the uniform file. [into] asks for a
    particular register, an operand that must sit in a group; it is not
    shared, and a register asked to hold two values is refused. So are a
    value that moves with two indices and one not affine in the one it
    moves with. *)
val value : t -> ?name:string -> cls:'k Core.vclass -> ?into:'k Core.reg -> ?base:some -> vars:(string * var) list -> Layouts.Expr.t -> 'k Core.reg

(** The kinds of register {!derive} computes into. *)
type _ dkind =
  | D_r : Core.r dkind
  | D_ur : Core.ur dkind
  | D_r2 : Core.r2 dkind

(** Instructions [code] computes into a fresh register of [kind] from fixed
    indices [over] and the registers [after], once where the last of them
    is set and shared by [key] from there. *)
val derive : t -> ?name:string -> ?after:Core.raw list -> kind:'k dkind -> over:var list -> key:string -> ('k Core.reg -> Core.stmt list) -> 'k Core.reg

(** {1 The finished program} *)

(** The program, every hole filled, each derived or construct instruction
    marked as such. *)
val finish : t -> Core.stmt list

(** The checks the program's instructions carry, by instruction. *)
val checks : t -> (int * Core.check) list

(** What the program's instructions touch, by instruction. *)
val touches : t -> (int * Core.touch) list

(** The warps {!warp_lane} named. *)
val warps : t -> int
