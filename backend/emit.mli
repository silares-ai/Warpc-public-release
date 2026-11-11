(** From an address expression to SASS.

   The expressions come from the layout algebra: a composite's strided form, or
   its pipeline with the compile-time coordinates pinned. The emitter chooses
   instructions and never decides what an address is. It knows the range of
   every variable, so it drops a division or a modulus the range makes exact --
   (32 l + 4 q) / 32 is l when 4 q < 32 -- and a subexpression it has emitted
   once is reused, not recomputed. *)

(** The smallest shift whose reciprocal divides exactly over [0, bound): the
   divisor is known at compile time, so this is a search, not an
   approximation, and it fails rather than round. *)
val exact_recip : int -> int -> int * int

(** the expression with its bindings substituted: sharing is recovered by the
   memo table, which sees the substituted copies as the same subexpression *)
val inline :
  (string * Layouts.Expr.t) list -> Layouts.Expr.t -> Layouts.Expr.t

(** A division or modulus by w of [whole + part], where w divides every
   coefficient of [whole], reads only [whole] when [part] lies in [0, w): the
   rewrite is exact over the ranges, not a guess. *)
val simplify : (string -> int) -> Layouts.Expr.t -> Layouts.Expr.t

type operand = Reg of int | Imm of int

type t = {
  b : Sass.t;
  mutable free : int list;
  gen : (unit -> int) option;
  memo : (Layouts.Expr.t, operand) Hashtbl.t;
  mutable high : int;
}

(** one emission site: scratch registers are handed out from [scratch] and are
   dead once the site's results have been read -- or, with [gen], each is a
   register [gen] declares, for the colouring to place *)
val create : ?gen:(unit -> int) -> Sass.t -> scratch:int list -> t

(** the value of [e] in register [dst]; [range] bounds every variable, [reg]
   names the register holding it *)
val into :
  t ->
  range:(string -> int) ->
  reg:(string -> int) -> Layouts.Expr.t -> dst:int -> unit
