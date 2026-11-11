(** Memory as the program sees it: buffers declared with their layouts, and
    operands that are a buffer at coordinates. No program writes an address.

    A buffer's layout takes its index coordinates -- a pipe's stage, a copy
    per warp, an accumulator buffer, a CTA's slot -- to a byte, and its
    element layout the coordinates inside one copy; an instruction's
    thread-value layout takes a lane and a value to element coordinates.
    The address an instruction needs is read off the composite: the strided
    form over the run-time indices is derived into registers
    ({!Prog.value}), the part a lane's position fixes is decided over the
    warp's lanes, and what the value index adds is the instruction's
    immediate.

    Every operand derived here carries a check: what it must hold where its
    instruction runs, computed from the buffer's layout at the coordinates
    -- not from the expression the derivation compiled -- which {!Interp}
    compares with what the instructions compute. *)

(** {1 Spaces and kinds}

    A buffer's space and kind are its type: a barrier operation takes a
    shared barrier, a load of a word a word of its space, an MMA a shared
    tile. *)

type shared = [ `Shared ]
type global = [ `Global ]

(** Copies with an element layout: an operand tile, a staging box. *)
type tile = [ `Tile ]

(** mbarriers. *)
type barrier = [ `Barrier ]

(** Words of a fixed size. *)
type word = [ `Word ]

(** Blocks of bytes an instruction's plan lays out: a tile's partial. *)
type slab = [ `Slab ]

(** A buffer of space ['s] and kind ['k]: its index modes and its copy's
    layout. *)
type ('s, 'k) buffer

(** A buffer at coordinates. *)
type ('s, 'k) term

(** A CTA's shared window, as this CTA addresses it; there is none of
    another space. *)
type 's window

(** {1 The kernel's memory} *)

(** A kernel's memory: its shared window, its place in the cluster, what it
    has declared. *)
type kernel

(** This CTA's window, and its rank in a cluster of [nx] x [ny] CTAs, x the
    fastest. A cluster of powers of two. *)
val create : ?cluster:int * int -> Prog.t -> kernel

val prog : kernel -> Prog.t

(** This CTA's rank in its cluster, the CTA field of its addresses. *)
val cta : kernel -> Core.ur Core.reg

(** The CTA's rank in its cluster, a fixed index. *)
val rank : kernel -> (Prog.fixed, Core.ur) Prog.index

type axis =
  | X
  | Y

(** The window of the cluster's CTA that agrees with this one except where
    [x] or [y] fixes a coordinate. *)
val peer : kernel -> ?x:int -> ?y:int -> unit -> shared window

(** The window of the cluster's CTA of rank [r]. *)
val peer_rank : kernel -> int -> shared window

(** The bytes a tile of layout [l] spans. *)
val footprint : ('a, 'b) Layouts.Layout.t -> elem:int -> int

(** A ring of [depth] copies of a shared tile, each aligned to the
    swizzle's period. Past shared memory is refused. *)
val ring : kernel -> name:string -> depth:int -> elem:int -> (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t -> (shared, tile) buffer

(** A block of shared memory held per warp or per slot: [modes] outermost
    first, each copy of layout [l]. *)
val shared : kernel -> name:string -> modes:int list -> elem:int -> (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t -> (shared, tile) buffer

(** [n] barriers of [count] arrivals, in the static window. *)
val barriers : kernel -> name:string -> n:int -> count:int -> (shared, barrier) buffer

(** [n] words of [bytes] each, in the static window. *)
val words : kernel -> name:string -> n:int -> bytes:int -> (shared, word) buffer

(** [n] words, or slabs, of [bytes] each in global memory, at the pointer
    [ptr]. *)
val global_words : name:string -> ptr:Core.r2 Core.reg -> n:int -> bytes:int -> (global, word) buffer

val global_slabs : name:string -> ptr:Core.r2 Core.reg -> n:int -> bytes:int -> (global, slab) buffer

(** {1 Operands} *)

(** A value over indices, for the operands that are coordinates: its
    expression, the indices it names, and its reference where it has one of
    its own. *)
type value

(** A coordinate of an index mode: fixed at compile time, an index of any
    kind, or a value of indices. *)
type coord =
  | C of int
  | I : ('k, 'c) Prog.index -> coord
  | V of value

(** [buf] at the coordinates of its index modes: this CTA's copy, or at
    [peer] another's. A coordinate outside its mode, an index that ranges
    past it, and a coordinate missing are refused where the term is used. *)
val at : ?peer:'s window -> ('s, 'k) buffer -> coord list -> ('s, 'k) term

(** The sub-tile of a tile's copy from row [r], column [c] on. *)
val sub : ('s, tile) term -> int * int -> ('s, tile) term

val idx : ('k, 'c) Prog.index -> value
val const : int -> value
val scale : int -> value -> value
val plus : value -> value -> value

(** An expression over indices; its reference, the expression's value. *)
val of_expr : Layouts.Expr.t -> (string * Prog.var) list -> value

(** A map's value at an index: its expression over "c", and the map for the
    reference. *)
val of_map : e:Layouts.Expr.t -> f:(int -> int) -> ('k, 'c) Prog.index -> value

(** This CTA's coordinate along an axis of its cluster. *)
val coord_of : kernel -> axis -> value

(** A value in a register of file [cls]: {!Prog.value} of its expression. *)
val get : kernel -> ?name:string -> cls:'c Core.vclass -> ?into:'c Core.reg -> value -> 'c Core.reg

(** A shared term's address in a register of file [cls]. *)
val addr : kernel -> cls:'c Core.vclass -> ?into:'c Core.reg -> ?within:value -> (shared, 'k) term -> 'c Core.reg

(** A term the shared window names by offset alone, as a request takes it;
    one moving with an index is refused. *)
val window_imm : (shared, 'k) term -> int

(** {1 Barriers} *)

(** A construct: every barrier the kernel declared, initialised with its
    arrival count. *)
val init_barriers : kernel -> unit

(** Until the barrier has completed the phase: a spin on one try-wait, at
    [parity] or the parity of the ring whose stage names the barrier. *)
val wait : kernel -> ?ns:int -> ?parity:Core.r Core.reg -> (shared, barrier) term -> unit

(** Ask whether the barrier has completed the phase, the answer in [into]
    later; the question reads its operands until it is answered. *)
val probe : kernel -> ?parity:Core.r Core.reg -> (shared, barrier) term -> into:Core.p Core.reg -> unit

val arrive_tx : kernel -> (shared, barrier) term -> tx:Core.r Core.reg -> unit

(** On another CTA's copy of the barrier: a reduction. *)
val arrive_red : kernel -> (shared, barrier) term -> unit

val arrive_tx_red : kernel -> (shared, barrier) term -> tx:Core.r Core.reg -> unit

(** The tensor core's commit, arriving once the MMAs ahead of it have run. *)
val commit : kernel -> (shared, barrier) term -> unit

(** The CTAs that agree with this one except along some axes, as a commit
    takes them. *)
type mask

(** The two-CTA commit, multicast to the CTAs of [mask]. *)
val commit2_mc : kernel -> leader:Core.up Core.reg -> (shared, barrier) term -> mask:mask -> unit

(** The CTAs that agree with this one except along [vary]. *)
val commit_mask : kernel -> axis list -> mask

(** {1 The copy engine and the tensor core} *)

(** A 2-D tensor-map box into the copy [dst], its bytes reported to [bar],
    its origin [x], [y] in the matrix. With [share], the box is shared by
    the cluster's CTAs along the axis: each loads its slice of the rows and
    multicasts it to them. *)
val tma_load :
  kernel -> ?two:bool -> ?share:axis -> map:Core.ur2 Core.reg -> dst:(shared, tile) term -> bar:(shared, barrier) term -> x:value -> y:value -> unit -> unit

(** The registers the MMAs take: the instruction descriptor, and two pairs
    of each operand's descriptor, which consecutive MMAs alternate between. *)
type mma_regs

val mma_regs :
  kernel -> atom:Atom.umma -> la:(Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t -> lb:(Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t -> mma_regs

(** One MMA: [a] and [b] its operands' sub-tiles, [d] the accumulator's
    address and the column its block starts at; [accumulate] the predicate
    that says whether it adds to what is there, or [`Always]. A two-CTA
    atom's MMA is issued by the pair's [leader]. A sub-tile off a swizzle
    row is refused. *)
val mma :
  kernel ->
  ?leader:Core.up Core.reg ->
  mma_regs ->
  a:(shared, tile) term ->
  b:(shared, tile) term ->
  d:Core.ur Core.reg * int ->
  accumulate:[ `Always | `If of Core.up Core.reg ] ->
  unit

(** A construct: tensor memory for [ncols] columns, its address in the
    register returned; [two] for the CTA pair. *)
val tmem_alloc : kernel -> ?two:bool -> ncols:int -> unit -> Core.ur Core.reg

val tmem_free : kernel -> base:Core.ur Core.reg -> ncols:int -> unit

(** {1 Words} *)

val lds : kernel -> Core.r Core.reg -> (shared, word) term -> unit
val sts : kernel -> (shared, word) term -> data:Core.r Core.reg -> unit
val lds128 : kernel -> Core.r4 Core.reg -> (shared, word) term -> unit

(** A request to the scheduler: its answer's 16 bytes into [answer], stated
    on [bar]. *)
val clc_request : kernel -> answer:(shared, word) term -> bar:(shared, barrier) term -> unit

(** A word of global memory past the L1, read and written. *)
val ldg_strong : kernel -> desc:Core.ur2 Core.reg -> Core.r Core.reg -> (global, word) term -> unit

val stg_strong : kernel -> desc:Core.ur2 Core.reg -> (global, word) term -> data:Core.r Core.reg -> unit

(** {1 The accumulator's way out, an instruction a call} *)

(** A warp's or a lane's index. *)
type lane_index = (Prog.fixed, Core.r) Prog.index

(** The accumulator at [acc] into the matrix at [mat], a general or a
    uniform pair, the tile's origin at [row], [col], as the plan reads it. *)
type direct

val direct :
  Store.direct ->
  acc:Core.ur Core.reg ->
  mat:[ `R of Core.r2 Core.reg | `UR of Core.ur2 Core.reg ] ->
  desc:Core.ur2 Core.reg ->
  row:value ->
  col:value ->
  warp:lane_index ->
  lane:lane_index ->
  direct

(** Block [(r, j)] of the accumulator into [data]: one tcgen05.ld. *)
val ldtm : kernel -> direct -> int * int -> Core.rv Core.reg -> unit

(** Register [v] of block [(r, j)], [data], to the matrix: one store. *)
val stg : kernel -> direct -> int * int -> int -> Core.r Core.reg -> unit

(** Registers [v] .. [v] + 3 of block [i] of a tile's partial at [slot], in
    fragment order: one load past the L1, one store. *)
val ldg_frag : kernel -> direct -> slot:(global, slab) term -> int -> int -> Core.rv Core.reg -> unit

val stg_frag : kernel -> direct -> slot:(global, slab) term -> int -> int -> Core.rv Core.reg -> unit

(** The accumulator at [acc] through [staging], a row of copies a lane
    quarter, to the matrix [map] names. *)
type staged

val staged :
  Store.staged ->
  staging:(shared, tile) buffer ->
  acc:Core.ur Core.reg ->
  map:Core.ur2 Core.reg ->
  row:value ->
  col:value ->
  warp:lane_index ->
  lane:lane_index ->
  staged

val ldtm_staged : kernel -> staged -> int * int -> Core.rv Core.reg -> unit

(** Vector [v] of a block's registers [data] into the warp's copy [slot]:
    one shared-memory store. *)
val sts_staged : kernel -> staged -> slot:int -> int -> Core.rv Core.reg -> unit

(** The warp's copy [slot] to block [(r, ch)]'s place in the matrix: one
    copy-engine store; [add], added to what the matrix holds. *)
val tma_store : ?add:bool -> kernel -> staged -> slot:int -> int * int -> unit

(** {1 The kernel} *)

(** The finished kernel: its launch read off the program -- the threads its
    warps take, the named barriers it meets at, whether it uses the tensor
    core, the shared memory and barriers declared here -- the declarations
    its body names, and every derived operand's check. *)
val finish :
  kernel ->
  name:string ->
  params:Core.param list ->
  grid:int * int ->
  tile:int * int * int ->
  ?together:int ->
  ?timing:Core.timing ->
  notes:string list ->
  unit ->
  Core.kernel
