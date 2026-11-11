(** The layouts the Blackwell instructions fix.

   A kernel never writes a layout. The instruction that consumes a tile fixes
   its layout, and every copy is compiled from the layouts of its two ends.
   This module holds those layouts as values of the algebra, and the deciders
   that read an instruction's encoding back off a layout. Each decider rebuilds
   the image the encoding describes and compares it with the layout at every
   coordinate, so a layout the instruction cannot express is refused here, at
   compile time, rather than read wrongly on the device. *)

(** the bytes of one swizzled row *)
val sw128_period : int

val at2 : ('a, 'b) Layouts.Layout.t -> int -> int -> int

(** [rows] rows of [cols] elements of [elem] bytes, each row exactly one
   swizzle span, rows packed, the 128-byte XOR over the whole. This is the
   image a SWIZZLE_128B tensor-map box produces and consumes, and the K-major
   operand a SWIZZLE_128B UMMA descriptor describes. *)
val swizzled_rows :
  rows:int ->
  cols:int ->
  elem:int ->
  (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t

type box = {
  box_rows : int;
  box_cols : int;
  box_elem : int;
  box_swizzle : int;
}

(** The tensor-map box whose image is [l]: a box of the tile's shape with the
   128-byte swizzle, required to agree with [l] everywhere. *)
val tma_box : ('a, 'b) Layouts.Layout.t -> elem:int -> box

type desc = { sbo : int; kstep : int; }

(** The K-major SWIZZLE_128B descriptor of [l]: rows one span apart inside an
   8-row group, groups SBO bytes apart, the XOR over the result. SBO is read
   off rows 0 and 8, which the swizzle leaves alone; the image the descriptor
   then describes is rebuilt and compared with [l]. The k steps of an MMA
   advance the start address, so every step's columns must be that image
   shifted by the step's offset before the swizzle -- also checked. *)
val umma_kmajor : ('a, 'b) Layouts.Layout.t -> elem:int -> mma_k:int -> desc

(** the descriptor's two words: start address and leading offset in the low
   word, SBO, version 1 and the layout type in the high one. LBO is not used
   by the swizzled K-major type; the encoding wants 1 there. *)
val desc_low_fixed : int

val desc_high : desc -> int

type elt = F16 | BF16 | F32

val elt_bytes : elt -> int

type major = K_major | MN_major

type src = Smem_desc | Tmem

type umma = {
  ab : elt;
  acc : elt;
  m : int;
  n : int;
  ctas : int;
  a_major : major;
  b_major : major;
  a_src : src;
}

(** the shapes CuTe's atoms accept for kind::f16 *)
val umma :
  ab:elt ->
  acc:elt ->
  m:int ->
  n:int -> ctas:int -> a_major:major -> b_major:major -> a_src:src -> umma

(** the K of one instruction: 256 bits of each operand row *)
val umma_k : umma -> int

val umma_a :
  umma ->
  (Layouts.Space.thread_value, Layouts.Space.logical) Layouts.Layout.t

val umma_b :
  umma ->
  (Layouts.Space.thread_value, Layouts.Space.logical) Layouts.Layout.t

val umma_c :
  umma ->
  (Layouts.Space.thread_value, Layouts.Space.logical) Layouts.Layout.t

(** The rows of an operand (or of the result) CTA [v] holds, read off the
   atom's layout: its first row and how many, refused unless they are one
   contiguous run of whole rows. *)
val cta_rows : ('a, 'b) Layouts.Layout.t -> cols:int -> v:int -> int * int

(** The 32-bit instruction descriptor, field by field as CuTe's
   UMMA::InstrDescriptor lays it out (cute/arch/mma_sm100_desc.hpp):
   c_format [4,6), a_format [7,10), b_format [10,13), a_major 15, b_major 16,
   N >> 3 at [17,23), M >> 4 at [24,29). Dense, no negation, no shift. *)
val idesc : umma -> int

(** the columns an allocation of at least [n] takes: a power of two, 32 at
   least -- what the allocator hands out *)
val tmem_columns : int -> int

(** The accumulator of a cta_group::1 M=128 MMA: row r is tensor-memory lane r,
   column c is column c, and an address is lane << 16 | column. *)
val tmem_accumulator :
  rows:int ->
  cols:int ->
  (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t

(** The accumulator a CTA holds, as the atom places it in tensor memory: row r
   of the CTA's share of the result, column c, to lane << 16 | column (CuTe's
   tmem_frg, cute/atom/mma_traits_sm100.hpp). With 128 rows a CTA -- M = 128
   on one CTA, M = 256 over a pair -- row r is lane r and column c column c.
   An M = 128 pair gives each CTA 64 rows, and CuTe's "2x2" atom for it puts
   the first half of the columns in lanes 0-63 and the second half in lanes
   64-127, (64, (N/2, 2)) : (1, (128, 64)) in its lane-major addressing, lane
   quarters 2 and 3 on output rows 64 on. The column mode is split in two digits, the half and
   the column within it; a row-major index decodes into them. *)
val tmem_acc :
  umma -> (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t

(** the columns of tensor memory the accumulator spans *)
val tmem_acc_columns : umma -> int

(** tcgen05.ld.32x32b.x[n], one warp's fragment: lane l receives lane l of the
   warp's 32, register r column r, over a [32 x n] block. The block's place in
   the accumulator is the warp's quarter of the lanes and the column chunk;
   the instruction fixes the quarter to the warp's index mod 4. *)
val ldtm_block : int

val ldtm_32x32b :
  n:int ->
  (Layouts.Space.thread_value, Layouts.Space.logical) Layouts.Layout.t

(** The instruction's contract on its composite into tensor memory. [at w ch
   l r] is the address the composite gives lane l's register r of block
   (w, ch), the block's place in the accumulator's division into 32-row by
   n-column blocks: one warp-uniform address per block, lane l at + l << 16
   and register r at + r from it, and the block inside one lane quarter --
   the warp whose index is that quarter mod 4 is the one that may load it. *)
val ldtm_quarter : at:('a -> 'b -> int -> int -> int) -> 'a -> 'b -> int

val check_ldtm :
  at:(int -> int -> int -> int -> int) ->
  blocks_w:int -> blocks_ch:int -> n:int -> unit

(** The slice of a tile one CTA of a multicast group loads: the first rows /
   group rows, as a copy-engine box, and where slice j starts. The slices must
   be that box's image at each start -- decided on every element. *)
val load_slice :
  (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t ->
  elem:int ->
  group:int ->
  (Layouts.Space.logical, Layouts.Space.physical) Layouts.Layout.t * 
  int * int
