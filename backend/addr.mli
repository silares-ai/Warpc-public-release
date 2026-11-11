(** Address arithmetic from layouts: an expression -- a composite's strided
   form, decided -- becomes the instructions that compute it into a declared
   register. Emit chooses the instructions; every intermediate value is a
   register declared here, which the colouring places like any other. *)

(** the value of [e] in [dst]; [range v] bounds variable v, [vars] names the
   register holding it *)
val into :
  range:(string -> int) ->
  vars:(string * Core.r Core.reg) list ->
  Layouts.Expr.t -> Core.r Core.reg -> Core.stmt list

(** A value over runtime coordinates [rt] and a compile-time index 0 .. n-1,
   as one expression in the runtime coordinates plus an immediate per index.
   The expression is the strided form of the map at index 0, decided; each
   immediate is read off by evaluation, and the split is refused unless the
   index shifts the value by the same amount at every runtime point. *)
val split :
  name:string ->
  rt:Layouts.Shape.t ->
  n:int -> (Layouts.Coord.t -> int -> int) -> Layouts.Expr.t * int array

(** the strided form of a map over [0, n) *)
val form : name:string -> n:int -> (int -> int) -> Layouts.Expr.t
