(** The accumulator's way out, as plans: what each warp loads from tensor
   memory and where each register goes, read off the layouts --
   the load's fragment composed with the accumulator's image, taken to output
   coordinates and through the matrix's layout. *)

(** [rows] x [cols] indices read as the index of the transposed [cols] x [rows] *)
val transpose :
  rows:int ->
  cols:int -> (Layouts.Space.logical, Layouts.Space.logical) Layouts.Layout.t

type direct = {
  ld : Layouts.Expr.t;
  imm_ld : int array;
  y : Layouts.Expr.t;
  imm_y : int array;
  x : Layouts.Expr.t;
  imm_x : int array;
  lane : Layouts.Expr.t;
  reg_off : int array;
  n : int;
  blocks : int;
  rep_ld : int array;
  rep_y : int array;
  rep_x : int array;
  cols : int;
  elem : int;
  ref_ld : int -> int * int -> int;
  ref_off : int -> int * int -> int -> int -> int;
}

(** [image]: the accumulator's tensor-memory layout, [trows] x [tcols] -- its
   lanes are output columns; [reps] such accumulators stacked [span] columns
   apart; the output matrix has [cols] columns of [elem] bytes. *)
val direct_plan :
  image:(Layouts.Space.logical, 'a) Layouts.Layout.t ->
  trows:int ->
  tcols:int -> reps:int -> span:int -> cols:int -> elem:int -> direct

(** the lane quarter a warp's tensor-memory loads are confined to *)
val quarter : Layouts.Expr.t -> Layouts.Expr.t

(** A buffer only the kernel reads back holds the accumulator in the order its
   stores leave the registers: block after block, a lane's registers four at
   a time, consecutive lanes' fours side by side -- so each 16-byte store and
   load of a warp is one contiguous 512 bytes. The layout is (block, vector,
   lane, element). *)
val frag_layout :
  direct -> (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t

val frag_at : direct -> blk:int -> lane:int -> reg:int -> int

(** this warp's slice of such a buffer: its lane quarter's blocks *)
val frag_lane : direct -> Layouts.Expr.t

type staged = {
  s_ld : Layouts.Expr.t;
  s_imm_ld : int array;
  s_copy : Layouts.Expr.t;
  s_y : Layouts.Expr.t;
  s_imm_y : int array;
  s_x : Layouts.Expr.t;
  s_imm_x : int array;
  sts : (Layouts.Space.thread_value, Layouts.Space.physical) Layouts.Layout.t;
  s_n : int;
  vec : int;
  s_elem : int;
  copy : int;
  copies : int;
  s_blocks : int;
  s_rep_ld : int array;
  s_rep_y : int array;
  s_rep_x : int array;
  s_ref_ld : int -> int * int -> int;
  ref_sts : int -> int -> int;
  ref_origin : int -> int * int -> int * int;
}

val staged_plan :
  image:(Layouts.Space.logical, 'a) Layouts.Layout.t ->
  trows:int ->
  tcols:int ->
  reps:int ->
  span:int ->
  lanes_are_cols:bool ->
  srows:int -> scols:int -> elem:int -> copies:int -> staged

(** the blocks a warp reads out, in order: every block of an accumulator
   before the next accumulator's *)
val blocks : direct -> (int * int) list

val staged_blocks : staged -> (int * int) list
