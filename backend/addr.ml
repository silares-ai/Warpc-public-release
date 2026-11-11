(* Address arithmetic from layouts: an expression -- a composite's strided
   form, decided -- becomes the instructions that compute it into a declared
   register. Emit chooses the instructions; every intermediate value is a
   register declared here, which the colouring places like any other. *)

open Layouts

(* the value of [e] in [dst]; [range v] bounds variable v, [vars] names the
   register holding it *)
let into ~range ~(vars : (string * Core.r Core.reg) list) (e : Expr.t) (dst : Core.r Core.reg) : Core.stmt list =
  let b = Sass.create () in
  let count = ref 0 in
  let em = Emit.create ~gen:(fun () -> incr count; Core.id (Core.r (Printf.sprintf "%s'%d" (Core.name dst) !count))) b ~scratch:[] in
  let reg v = match List.assoc_opt v vars with Some r -> Core.id r | None -> failwith ("Addr.into: unbound " ^ v) in
  Emit.into em ~range ~reg e ~dst:(Core.id dst);
  Sass.items b

(* A value over runtime coordinates [rt] and a compile-time index 0 .. n-1,
   as one expression in the runtime coordinates plus an immediate per index.
   The expression is the strided form of the map at index 0, decided; each
   immediate is read off by evaluation, and the split is refused unless the
   index shifts the value by the same amount at every runtime point. *)
let split ~name ~rt ~n f =
  let pts = Coord.enumerate rt in
  let e =
    match Decide.strided_form ~shape:rt ~offset:(fun c -> f c 0) with
    | Some e -> e
    | None -> failwith (name ^ ": no strided form over the runtime coordinates")
  in
  let imm k =
    match List.sort_uniq compare (List.map (fun c -> f c k - f c 0) pts) with
    | [ d ] -> d
    | _ -> failwith (name ^ ": the compile-time index does not shift the value by a constant")
  in
  e, Array.init n imm

(* the strided form of a map over [0, n) *)
let form ~name ~n f =
  match Decide.strided_form ~shape:(Shape.Bound n) ~offset:(function Coord.Idx t -> f t | Tuple _ -> assert false) with
  | Some e -> e
  | None -> failwith (name ^ ": no strided form")
