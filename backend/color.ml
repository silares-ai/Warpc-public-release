(* Register colouring for the core language: liveness over the program's
   control flow, an interference graph for each register class, and a greedy
   colouring in the order the registers are first written. A register that
   does not fit is an error that names the point of greatest pressure and
   what is live there. Nothing is spilled.

   The classes and what they give a program: R0-R252 without R1, the stack
   pointer; UR0-UR62, UR63 being URZ; P0-P6 and UP0-UP6, the seventh being
   PT and UPT. *)

open Core
module IS = Set.Make (Int)

let limit = function R -> 253 | UR -> 63 | P -> 7 | UP -> 7
let reserved = function R -> [ 1 ] | _ -> []

(* the declared register a unit of an operand belongs to, by [lookup] *)
let owner lookup = function
  | Sass.R u | UR u | P u | UP u when u >= first_placeholder ->
    (match lookup u with Some r -> Some r | None -> failwith (Printf.sprintf "placeholder %d undeclared" u))
  | _ -> None

let unit_of = function Sass.R u | UR u | P u | UP u -> u

let describe (ins : Sass.insn) = Sass.to_string ins

(* Liveness of each register unit -- a pair's halves are written apart -- at
   each instruction, backwards to a fixed point. A guarded write kills
   nothing. *)
let liveness lookup (arr : Sass.item array) =
  let n = Array.length arr in
  let label_index = Hashtbl.create 16 in
  Array.iteri (fun i it -> match it with Sass.Label l -> Hashtbl.replace label_index l i | I _ -> ()) arr;
  let target l = match Hashtbl.find_opt label_index l with Some i -> i | None -> failwith ("branch to no label " ^ l) in
  let succs i =
    match arr.(i) with
    | Sass.Label _ -> if i + 1 < n then [ i + 1 ] else []
    | I ins ->
      let fall = if i + 1 < n then [ i + 1 ] else [] in
      (match ins.branch with
       | `No -> fall
       | `Exit -> if ins.guard = None then [] else fall
       | `Bra l -> if ins.guard = None then [ target l ] else target l :: fall)
  in
  let units rs = List.fold_left (fun s r -> match owner lookup r with Some _ -> IS.add (unit_of r) s | None -> s) IS.empty rs in
  let uses = Array.make n IS.empty and kills = Array.make n IS.empty and defs = Array.make n IS.empty in
  let late = Array.make n IS.empty in
  Array.iteri
    (fun i it ->
      match it with
      | Sass.Label _ -> ()
      | I ins ->
        uses.(i) <- units (ins.uses @ ins.late_uses);
        late.(i) <- units ins.late_uses;
        defs.(i) <- units ins.defs;
        if ins.guard = None then kills.(i) <- defs.(i))
    arr;
  let succ = Array.init n succs in
  let live_in = Array.make n IS.empty and live_out = Array.make n IS.empty in
  let changed = ref true in
  while !changed do
    changed := false;
    for i = n - 1 downto 0 do
      let out = List.fold_left (fun s j -> IS.union s live_in.(j)) IS.empty succ.(i) in
      let inn = IS.union uses.(i) (IS.diff out kills.(i)) in
      if not (IS.equal out live_out.(i)) then (live_out.(i) <- out; changed := true);
      if not (IS.equal inn live_in.(i)) then (live_in.(i) <- inn; changed := true)
    done
  done;
  (* A unit an asynchronous unit reads late holds its value until the
     program writes it again: the read may come after the instruction that
     would otherwise reuse the register. So from a late read on, the unit is
     pending until its next write, forwards along the control flow. *)
  let preds = Array.make n [] in
  Array.iteri (fun i ss -> List.iter (fun j -> preds.(j) <- i :: preds.(j)) ss) succ;
  let pend_out = Array.make n IS.empty in
  let changed = ref true in
  while !changed do
    changed := false;
    for i = 0 to n - 1 do
      let inn = List.fold_left (fun s j -> IS.union s pend_out.(j)) IS.empty preds.(i) in
      let out = IS.union late.(i) (IS.diff inn kills.(i)) in
      if not (IS.equal out pend_out.(i)) then (pend_out.(i) <- out; changed := true)
    done
  done;
  let live_out = Array.mapi (fun i s -> IS.union s pend_out.(i)) live_out in
  live_in, live_out, defs, uses

(* the colouring: each declared register's first physical register. [owner]
   finds the declaration of a placeholder unit: a kernel's own table, or
   while a program is being built, the declarations so far. *)
let color ?(owner = Core.owner) (items : Sass.item list) : (int, int) Hashtbl.t =
  let reg_of u = match owner u with Some r -> r | None -> failwith (Printf.sprintf "placeholder %d undeclared" u) in
  let owners_of s = List.sort_uniq compare (List.map (fun u -> (reg_of u).id) (IS.elements s)) in
  let names ids = String.concat " " (List.map (fun id -> (reg_of id).name) ids) in
  let arr = Array.of_list items in
  let n = Array.length arr in
  let live_in, live_out, defs, uses = liveness owner arr in
  (* a register read before anything writes it, on some path from the entry *)
  (match n with
   | 0 -> ()
   | _ ->
     if not (IS.is_empty live_in.(0))
     then
       failwith
         (Printf.sprintf "registers read before they are written on some path from the entry: %s"
            (names (owners_of live_in.(0)))));
  let adj : (int, IS.t) Hashtbl.t = Hashtbl.create 256 in
  let edge a b =
    if a <> b && (reg_of a).cls = (reg_of b).cls then begin
      Hashtbl.replace adj a (IS.add b (Option.value (Hashtbl.find_opt adj a) ~default:IS.empty));
      Hashtbl.replace adj b (IS.add a (Option.value (Hashtbl.find_opt adj b) ~default:IS.empty))
    end
  in
  let first_def = Hashtbl.create 256 in
  Array.iteri
    (fun i it ->
      match it with
      | Sass.Label _ -> ()
      | I _ ->
        let ds = owners_of defs.(i) in
        List.iter
          (fun d ->
            if not (Hashtbl.mem first_def d) then Hashtbl.replace first_def d i;
            (* what a write must not clobber: everything live after it, the
               other results of the instruction, and -- for a wide operand --
               the sources it is still reading *)
            List.iter (edge d) (owners_of live_out.(i));
            List.iter (edge d) ds;
            List.iter (fun u -> if (reg_of d).width > 1 || (reg_of u).width > 1 then edge d u) (owners_of uses.(i)))
          ds)
    arr;
  (* every register used anywhere, written or not *)
  Array.iter (fun s -> List.iter (fun u -> if not (Hashtbl.mem first_def u) then Hashtbl.replace first_def u max_int) (owners_of s)) uses;
  let order = List.sort (fun (a, i) (b, j) -> compare (i, a) (j, b)) (Hashtbl.fold (fun id i acc -> (id, i) :: acc) first_def []) in
  let base = Hashtbl.create 256 in
  (* the point of greatest pressure in a class, for the error *)
  let pressure cls =
    let best = ref (-1, 0, []) in
    Array.iteri
      (fun i s ->
        let live = List.filter (fun u -> (reg_of u).cls = cls) (IS.elements s) in
        let units = List.length live in
        let _, u, _ = !best in
        if units > u then best := i, units, owners_of (IS.of_list live))
      live_out;
    !best
  in
  List.iter
    (fun (id, _) ->
      let x = reg_of id in
      let taken = Hashtbl.create 64 in
      List.iter (fun r -> Hashtbl.replace taken r ()) (reserved x.cls);
      IS.iter
        (fun nb ->
          match Hashtbl.find_opt base nb with
          | Some b -> for k = 0 to (reg_of nb).width - 1 do Hashtbl.replace taken (b + k) () done
          | None -> ())
        (Option.value (Hashtbl.find_opt adj id) ~default:IS.empty);
      let fits b = let ok = ref true in for k = 0 to x.width - 1 do if Hashtbl.mem taken (b + k) then ok := false done; !ok in
      let rec find b = if b + x.width > limit x.cls then None else if fits b then Some b else find (b + x.align) in
      match find 0 with
      | Some b -> Hashtbl.replace base id b
      | None ->
        let i, units, live = pressure x.cls in
        let at = match arr.(i) with Sass.I ins -> describe ins | Label l -> l ^ ":" in
        failwith
          (Printf.sprintf
             "%s registers: %s (%d wide) does not fit. The most live at once is %d, after instruction %d (%s): %s"
             (cls_name x.cls) x.name x.width units i at (names live)))
    order;
  base
