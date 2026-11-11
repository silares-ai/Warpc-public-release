(* A program under construction: statements appended one call at a time, so
   a program reads as a sequence of lines, each one instruction. Two kinds of
   instruction are not the program's lines, and each is marked where it
   lands. The control of the language's constructs -- a loop's count,
   compare and branch; a ring's step; a role's dispatch; a spin's sleep and
   branch -- is emitted by the construct. The address arithmetic a layout
   implies goes into holes, filled when the program is finished, so a
   derived instruction lands where it belongs whatever order it was asked
   for in:
   - a value of indices that stay fixed is computed once, where the last of
     them (or the base register it adds to) is set, and shared from there;
   - a value affine in a counter is set where the loop starts and stepped
     where it steps;
   - a value affine in a ring position is one multiply-add from the stage,
     where the stage is asked about at the top of an iteration, ahead of the
     wait that follows. *)

open Layouts
open Core

(* the kinds of register a derived value is kept in, with a witness the
   caches compare *)
type _ dkind =
  | D_r : r dkind
  | D_ur : ur dkind
  | D_r2 : r2 dkind

type (_, _) eq = Refl : ('a, 'a) eq

let same : type a b. a dkind -> b dkind -> (a, b) eq option =
 fun a b -> match a, b with D_r, D_r -> Some Refl | D_ur, D_ur -> Some Refl | D_r2, D_r2 -> Some Refl | _ -> None

let dkind_of : type k. k vclass -> k dkind = function General -> D_r | Uniform -> D_ur

(* a register with its kind *)
type held = Held : 'k dkind * 'k reg -> held

type item =
  | S of stmt
  | H of hole

and hole =
  { seq : int (* program order *)
  ; kind : Sass.origin (* what its instructions are: derived, or a ring's step *)
  ; mutable fill : stmt list
  ; mutable cache : (string * held) list (* values computed here, by key *)
  }

let cached : type k. hole -> k dkind -> string -> k reg option =
 fun h k key -> match List.assoc_opt key h.cache with Some (Held (k', x)) -> (match same k' k with Some Refl -> Some x | None -> None) | None -> None

(* a register of the general or the uniform file, with its file *)
type some = Some_reg : 'c vclass * 'c reg -> some

let some_id (Some_reg (_, x)) = id x

type fixed = [ `Fixed ]
type counter = [ `Counter ]
type ring = [ `Ring ]

(* A run-time index: a register with a range. [Fixed] holds one value from
   where it is set to the end of the scope; [Counter] starts at zero and
   steps by one; [Ring] steps by one and wraps at [depth], flipping [parity]. *)
type kind =
  | Fixed
  | Counter
  | Ring of ring_state

and ring_state =
  { depth : int
  ; stays : up reg (* this step does not wrap *)
  ; stays_p : p reg (* the same as a P *)
  ; parity : r reg (* the phase a wait on this stage's barrier waits for *)
  ; mutable now : hole option (* from [ask] to [advance]: where this iteration's values of it go *)
  }

type ('k, 'c) index =
  { name : string
  ; ireg : 'c reg
  ; icls : 'c vclass
  ; range : int
  ; kind : kind
  ; at : hole (* where it is set: fixed values of it are computed here, inductions start here *)
  ; mutable steps : hole list
  ; mutable inductions : (some * int) list (* a register following it, and its step *)
  }

(* an index of any kind, for the variables of a value *)
type var = Var : ('k, 'c) index -> var

type t =
  { mutable items : item list (* reversed *)
  ; mutable next_seq : int
  ; mutable guard : (stmt -> stmt) option
  ; mutable indices : var list
  ; mutable roles : hole list (* innermost first: where a role's constants are computed *)
  ; mutable sets : (int * hole) list (* a register set inside the program, and the hole after it *)
  ; mutable intos : (int * string) list (* a register a value was asked into, and that value *)
  ; mutable checks : (int * check) list (* what a derived operand must hold, by instruction *)
  ; mutable touches : (int * touch) list (* what an instruction touches its registers do not say *)
  ; mutable warps : int option
  }

let new_hole ?(kind = Sass.Derived) p =
  let h = { seq = p.next_seq; kind; fill = []; cache = [] } in
  p.next_seq <- p.next_seq + 1;
  p.items <- H h :: p.items;
  h

let create () = { items = []; next_seq = 0; guard = None; indices = []; roles = []; sets = []; intos = []; checks = []; touches = []; warps = None }

let guarded p s = match p.guard with None -> s | Some g -> g s
let mark o (s : stmt) = match s with Sass.I ins -> Sass.I { ins with origin = o } | l -> l

(* the registers statements set: a value added to one is computed after its
   last setting *)
let note_defs p (l : stmt list) =
  let owner_id u = Option.map (fun (o : raw) -> o.id) (owner u) in
  let ids = List.concat_map (function Sass.I ins -> List.filter_map (function Sass.R u | Sass.UR u -> owner_id u | _ -> None) ins.Sass.defs | Label _ -> []) l in
  if ids <> [] then begin
    let h = new_hole p in
    List.iter (fun id -> p.sets <- (id, h) :: p.sets) (List.sort_uniq compare ids)
  end

let emit p s =
  p.items <- S (guarded p s) :: p.items;
  note_defs p [ s ]

let emits p l = List.iter (emit p) l

(* an instruction of a construct: the language's, not the program's line *)
let construct p s =
  p.items <- S (mark Sass.Construct (guarded p s)) :: p.items;
  note_defs p [ s ]

let constructs p l = List.iter (construct p) l

(* what the operands of [s] must hold where it runs *)
let check p (s : stmt) cs = match s with Sass.I ins -> p.checks <- List.map (fun c -> ins.id, c) cs @ p.checks | Label _ -> ()

(* what [s] touches that its registers do not say *)
let touch p (s : stmt) ts = match s with Sass.I ins -> p.touches <- List.map (fun t -> ins.id, t) ts @ p.touches | Label _ -> ()

(* the program's instruction [s], its derived operands checked by [cs] *)
let emit_checked p s cs =
  check p s cs;
  emit p s

let fill h l = h.fill <- h.fill @ l
let label p l = p.items <- S (Core.label l) :: p.items

(* instructions the compiler derives, here: an operand's address formed
   where the instruction takes it *)
let derived p l =
  let h = new_hole p in
  h.fill <- l;
  note_defs p l

let fresh = Core.fresh_label

(* A role: the warps that run [f], its constants computed at its start.
   With [skip], a compare into the predicate it is given: the warps it
   holds for branch past the role. *)
let role p ?(name = "role") ?skip f =
  let past =
    Option.map
      (fun cmp ->
        let q = Core.p "role" and l = fresh ("NOT_" ^ String.uppercase_ascii name) in
        construct p (cmp q);
        construct p (Core.bra q l);
        l)
      skip
  in
  p.roles <- new_hole p :: p.roles;
  let r = f () in
  p.roles <- List.tl p.roles;
  Option.iter (label p) past;
  r

(* every instruction [f] emits, under the predicate *)
let when_ p ?(neg = false) g f =
  if p.guard <> None then failwith "Prog.when_: an instruction takes one guard";
  p.guard <- Some (Core.on ~neg g);
  f ();
  p.guard <- None

let step_by (Some_reg (c, d)) k : stmt = match c with Uniform -> Core.uiadd3 d d k | General -> Core.iadd3_c d d k

let close_index (Var i) =
  let ds = List.rev i.inductions in
  List.iter
    (fun h ->
      match i.kind with
      | Fixed -> ()
      | Counter -> fill h (List.map (fun (d, k) -> step_by d k) ds)
      | Ring r when r.depth = 1 -> ()
      | Ring r ->
        (* the steps a row apart from the wind-backs; a uniform instruction
           takes the uniform predicate, a general one the general *)
        let on_stays ?neg (Some_reg (c, _) as d) k = match c with Uniform -> Core.on ?neg r.stays (step_by d k) | General -> Core.on ?neg r.stays_p (step_by d k) in
        fill h (List.map (fun (d, k) -> on_stays d k) ds @ List.map (fun (d, k) -> on_stays ~neg:true d (-(r.depth - 1) * k)) ds))
    i.steps

(* the program, every hole filled, each derived or construct instruction
   marked as such *)
let finish p =
  List.iter close_index p.indices;
  List.concat_map (function S s -> [ s ] | H h -> List.map (mark h.kind) h.fill) (List.rev p.items)

let checks p = List.rev p.checks
let touches p = List.rev p.touches
let warps p = match p.warps with Some w -> w | None -> failwith "Prog: the program never named its warps (Prog.warp_lane)"

(* A spin: [test] -- the instruction that sets the predicate it is given
   when what it waits for has happened -- then, until it has, a sleep and a
   branch back. *)
let spin p ?ns ?(checks = []) test =
  let q = Core.p "waited" and l = fresh "WAIT" in
  label p l;
  emit_checked p (test q) checks;
  construct p (Core.on ~neg:true q (Core.sleep_syncs ?ns ()));
  construct p (Core.bra ~neg:true q l)

(* ---- indices ---- *)

let make p ~name ~reg ~cls ~range ~kind ~inductions =
  let i = { name; ireg = reg; icls = cls; range; kind; at = new_hole p; steps = []; inductions } in
  p.indices <- Var i :: p.indices;
  i

let index_name i = i.name
let index_range i = i.range
let reg i = i.ireg

(* an index the program has just set, fixed from here to the end of the
   scope it was set in *)
let fixed p ~name ~range ~cls reg : (fixed, _) index = make p ~name ~reg ~cls ~range ~kind:Fixed ~inductions:[]

(* A loop over the values [reg] takes: the index is fixed for an iteration,
   its values computed at the top of each; [next] sets the next value and
   branches back to [top]. *)
let over p ~name ~reg ~range ~next body =
  let top = fresh "TILES" in
  label p top;
  body (make p ~name ~reg ~cls:General ~range ~kind:Fixed ~inductions:[] : (fixed, r) index);
  next top

(* a persistent loop: from what [reg] holds, [step] at a time while below
   [until], the first always taken *)
let stride p ~name ?pred ~reg ~until ~step body =
  let pred = match pred with Some q -> q | None -> Core.p (name ^ "_left") in
  over p ~name ~reg ~range:until body ~next:(fun top ->
    construct p (Core.iadd3_c reg reg step);
    construct p (Core.isetp_lt_u32 pred reg until);
    construct p (Core.bra pred top))

(* this thread's warp and lane, from its index [tid] in the CTA *)
let warp_lane p ~tid ~warps =
  p.warps <- Some warps;
  let warp = Core.r "warp" and lane = Core.r "lane" in
  derived p [ Core.shf_r warp tid 5; Core.lop3_and lane tid 31 ];
  fixed p ~name:"warpid" ~range:warps ~cls:General warp, fixed p ~name:"laneid" ~range:32 ~cls:General lane

type count =
  | Times of int
  | Reg_times of r reg

(* for i = 0 .. count - 1, at least once. The count and the compare come
   first in the body, so the compare's latency runs under it. *)
let for_ p ~name ?pred ~count body =
  let pred = match pred with Some q -> q | None -> Core.p (name ^ "_more") in
  let c = Core.r name in
  let range = match count with Times n -> n | Reg_times _ -> 1 lsl 20 in
  construct p (Core.mov_rz c);
  let i : (counter, r) index = make p ~name ~reg:c ~cls:General ~range ~kind:Counter ~inductions:[] in
  let top = fresh "LOOP" in
  label p top;
  construct p (Core.iadd3_c c c 1);
  construct p (match count with Times n -> Core.isetp_lt_u32 pred c n | Reg_times n -> Core.isetp_rr ~cmp:"LT" pred c n);
  body i;
  i.steps <- new_hole p :: i.steps;
  construct p (Core.bra pred top)

(* a ring position: the stage, a uniform register, and the parity of the
   phase its barriers are waited for at *)
let ring p ~name ~depth ~parity : (ring, ur) index =
  let stage = Core.ur (name ^ "_stage") and par = Core.r (name ^ "_parity") in
  let rg = { depth; stays = Core.up (name ^ "_stays"); stays_p = Core.p (name ^ "_stays_p"); parity = par; now = None } in
  construct p (Core.umov stage 0);
  construct p (Core.mov_imm par parity);
  make p ~name ~reg:stage ~cls:Uniform ~range:depth ~kind:(Ring rg) ~inductions:[ Some_reg (Uniform, stage), 1 ]

let ring_of i = match i.kind with Ring r -> r | _ -> failwith (i.name ^ ": not a ring")

(* the parity a wait on a barrier named by this index's stage waits for *)
let ring_parity (Var i) = match i.kind with Ring r -> Some r.parity | _ -> None

(* the top of an iteration: whether the next step wraps, and this stage's
   addresses, asked as early as the stage is known *)
let ask p i =
  let r = ring_of i in
  if r.now <> None then failwith (i.name ^ ": asked twice before advancing");
  construct p (Core.uisetp_ne_imm r.stays i.ireg (r.depth - 1));
  r.now <- Some (new_hole p)

(* the step, and the parity when it wraps *)
let advance p i =
  let r = ring_of i in
  if r.now = None then failwith (i.name ^ ": advanced before it was asked");
  r.now <- None;
  construct p (Core.plop3_up r.stays_p r.stays);
  i.steps <- new_hole ~kind:Sass.Construct p :: i.steps;
  construct p (Core.on ~neg:true r.stays_p (Barrier.flip r.parity))

(* One step of a ring: [body] runs at the current stage -- its values of
   the ring computed at the top, whether this step wraps asked there too --
   and the ring moves to the next stage after it, flipping its parity when
   it wraps. *)
let step p (i : (ring, ur) index) body =
  ask p i;
  let r = body () in
  advance p i;
  r

(* an index's value where a check runs: a counter's register is one ahead
   inside its loop *)
let index_value (env : env) i = match i.kind with Counter -> env.value i.ireg - 1 | _ -> env.value i.ireg
let var_value env (Var i) = index_value env i
let var_name (Var i) = i.name
let var_range (Var i) = i.range

(* ---- derived values ---- *)

let rec vars_of (e : Expr.t) =
  match e with
  | Const _ -> []
  | Var v -> [ v ]
  | Sum (ts, _) -> List.concat_map (fun (_, t) -> vars_of t) ts
  | Div (x, _) | Mod (x, _) -> vars_of x
  | Xor (a, b) -> vars_of a @ vars_of b
  | Let (x, a, b) -> vars_of a @ List.filter (( <> ) x) (vars_of b)

(* e = k * v + rest, rest free of v, when e is affine in v *)
let split_on (e : Expr.t) v =
  if not (List.mem v (vars_of e)) then Some (0, e)
  else
    match e with
    | Var _ -> Some (1, Expr.Const 0)
    | Sum (ts, c) ->
      (match List.partition (fun (_, t) -> List.mem v (vars_of t)) ts with
       | [ (k, Var _) ], [] -> Some (k, Expr.Const c)
       | [ (k, Var _) ], rest -> Some (k, Expr.Sum (rest, c))
       | _ -> None)
    | _ -> None

(* a register of file [want] holding [x]: itself, or a copy *)
let to_cls : type k c. k vclass -> c vclass -> c reg -> stmt list * k reg =
 fun want have x ->
  match want, have with
  | Uniform, Uniform -> [], x
  | General, General -> [], x
  | Uniform, General -> let u = Core.ur (Core.name x ^ "_u") in [ Core.r2ur u x ], u
  | General, Uniform -> let g = Core.r (Core.name x ^ "_r") in [ Core.mov_ur g x ], g

let var_reg (vars : (string * var) list) v = match List.assoc_opt v vars with Some (Var i) -> Some_reg (i.icls, i.ireg) | None -> failwith ("Prog.compute: unbound " ^ v)

(* [e] + [base] into [d]: an affine [e] as a chain of multiply-adds in [d]'s
   file, anything else through Emit on the general registers *)
let compute : type k. vars:(string * var) list -> ?base:some -> Expr.t -> k vclass -> k reg -> stmt list =
 fun ~vars ?base e cls d ->
  let conv (Some_reg (c, x)) = to_cls cls c x in
  let add (a : k reg) (x : k reg) : stmt = match cls with Uniform -> Core.uiadd3_uu d a x | General -> Core.iadd3 d a x in
  let with_base = match base with None -> [] | Some b -> let pre, b = conv b in pre @ [ add d b ] in
  if Expr.is_affine e then begin
    let ts, c = match e with Const c -> [], c | Sum (ts, c) -> ts, c | Var v -> [ 1, Expr.Var v ], 0 | _ -> assert false in
    (* the base first, then each term multiplied in, then the constant *)
    let (start : k reg option), pre = match base with None -> None, [] | Some b -> let pre, b = conv b in Some b, pre in
    let (acc : k reg option), code =
      List.fold_left
        (fun ((acc : k reg option), code) (k, t) ->
          let x = match t with Expr.Var v -> var_reg vars v | _ -> assert false in
          let p1, x = conv x in
          let s : stmt =
            match cls, acc, k with
            | Uniform, None, 1 -> Core.uiadd3 d x 0
            | Uniform, None, _ -> Core.uimad_z d x k
            | Uniform, Some a, 1 -> Core.uiadd3_uu d a x
            | Uniform, Some a, _ -> Core.uimad_imm d x k a
            | General, None, 1 -> Core.mov_rr d x
            | General, None, _ -> Core.imad_rz d x k
            | General, Some a, 1 -> Core.iadd3 d a x
            | General, Some a, _ -> Core.imad d x k a
          in
          Some d, code @ p1 @ [ s ])
        (start, pre) ts
    in
    code
    @
    match acc, cls with
    | None, Uniform -> [ Core.umov d c ]
    | None, General -> [ Core.mov_imm d c ]
    | Some a, _ when c = 0 -> if Core.id a = Core.id d then [] else [ (match cls with Uniform -> Core.uiadd3 d a 0 | General -> Core.mov_rr d a) ]
    | Some a, Uniform -> [ Core.uiadd3 d a c ]
    | Some a, General -> [ Core.iadd3_c d a c ]
  end
  else begin
    let g : r reg = match cls with General -> d | Uniform -> Core.r (Core.name d ^ "_r") in
    let used = List.sort_uniq compare (vars_of e) in
    let conv_r = List.map (fun v -> let (Some_reg (c, x)) = var_reg vars v in let pre, x = to_cls General c x in v, pre, x) used in
    let range v = match List.assoc_opt v vars with Some (Var i) -> i.range | None -> 1 in
    List.concat_map (fun (_, pre, _) -> pre) conv_r
    @ Addr.into ~range ~vars:(List.map (fun (v, _, x) -> v, x) conv_r) e g
    @ (match cls with Uniform -> [ Core.r2ur d g ] | General -> [])
    @ with_base
  end

(* The value of [e] + [base] in a register of file [cls], [e] over the
   indices [vars] names. A value of fixed indices is computed once, where
   the last of them is set -- or, of none, where the role starts -- and
   shared from there. A value affine in a counter or a ring follows it: set
   where it starts, stepped where it steps. [into] asks for a particular
   register, an operand that must sit in a group; it is not shared. *)
let rec value : type k. t -> ?name:string -> cls:k vclass -> ?into:k reg -> ?base:some -> vars:(string * var) list -> Expr.t -> k reg =
 fun p ?(name = "addr") ~cls ?into ?base ~vars e ->
  let used = List.sort_uniq compare (vars_of e) in
  let idx v = match List.assoc_opt v vars with Some i -> i | None -> failwith ("Prog.value: unbound " ^ v) in
  let fixed_holes = List.filter_map (fun v -> let (Var i) = idx v in if i.kind = Fixed then Some i.at else None) used in
  let base_hole = match base with Some b -> Option.to_list (List.assoc_opt (some_id b) p.sets) | None -> [] in
  (* the latest of the role's start, the fixed indices' and the base's
     settings; outside a role and of none, where the value is asked for *)
  let deps = fixed_holes @ base_hole @ match p.roles with h :: _ -> [ h ] | [] -> [] in
  let latest = List.fold_left (fun a h -> match a with Some g when g.seq >= h.seq -> a | _ -> Some h) None deps in
  let home () = match latest with Some h -> h | None -> new_hole p in
  let fresh_reg n : k reg = match into with Some d -> d | None -> Core.fresh_in cls n in
  let key = Printf.sprintf "%s|%s|%s" (cls_name (cls_of_vclass cls)) (match base with Some b -> string_of_int (some_id b) | None -> "-") (Expr.to_string e) in
  (* computed once in hole [h], shared there unless it goes to [into] *)
  let once h name code : k reg =
    match into, cached h (dkind_of cls) key with
    | None, Some d -> d
    | _ ->
      let d = fresh_reg name in
      (match into with
       | Some d ->
         (match List.assoc_opt (Core.id d) p.intos with
          | Some k when k <> key -> failwith (Printf.sprintf "Prog.value: %s already holds %s" (Core.name d) k)
          | _ -> p.intos <- (Core.id d, key) :: p.intos)
       | None -> ());
      fill h (code d);
      if into = None then h.cache <- (key, Held (dkind_of cls, d)) :: h.cache;
      p.sets <- (Core.id d, h) :: p.sets;
      d
  in
  let affine v = match split_on e v with Some kr -> kr | None -> failwith (Printf.sprintf "Prog.value: %s is not affine in %s" (Expr.to_string e) v) in
  match List.filter (fun v -> let (Var i) = idx v in i.kind <> Fixed) used with
  | [] -> once (home ()) name (compute ~vars ?base e cls)
  | [ v ] ->
    let (Var i) = idx v in
    let k, rest = affine v in
    (match i.kind with
     | Fixed -> assert false
     | Counter ->
       if (match latest with Some h -> h.seq > i.at.seq | None -> false) then failwith (Printf.sprintf "Prog.value: %s needs a value set after %s starts" (Expr.to_string e) i.name);
       let d = fresh_reg (name ^ "@" ^ i.name) in
       fill i.at (compute ~vars ?base rest cls d);
       i.inductions <- (Some_reg (cls, d), k) :: i.inductions;
       d
     | Ring r ->
       (* inside an iteration, at its top; between iterations, where it is
          asked, from the stage the ring has stepped to *)
       let now = match r.now with Some h -> h | None -> new_hole p in
       (match cls, i.icls with
        | Uniform, Uniform ->
          (* the stage times its step, plus the rest: a register when it is
             the base alone, a value of the fixed indices otherwise *)
          let addend : ur reg =
            match rest, base with
            | Expr.Const 0, Some (Some_reg (Uniform, b)) -> b
            | _ -> value p ~name:(name ^ "_base") ~cls:Uniform ?base ~vars rest
          in
          once now (name ^ "@" ^ i.name) (fun d -> [ (if k = 1 then Core.uiadd3_uu d i.ireg addend else Core.uimad_imm d i.ireg k addend) ])
        | _ -> failwith (Printf.sprintf "Prog.value: %s is a uniform value of %s" (Expr.to_string e) i.name)))
  | _ -> failwith (Printf.sprintf "Prog.value: %s moves with more than one index" (Expr.to_string e))

(* Instructions [code] computes into a fresh register of kind [kind] from
   fixed indices [over] and the registers [after], once where the last of
   them is set and shared by [key] from there. *)
let derive : type k. t -> ?name:string -> ?after:raw list -> kind:k dkind -> over:var list -> key:string -> (k reg -> stmt list) -> k reg =
 fun p ?(name = "derived") ?(after = []) ~kind ~over ~key code ->
  List.iter (fun (Var i) -> if i.kind <> Fixed then failwith ("Prog.derive: " ^ i.name ^ " moves")) over;
  let holes = List.map (fun (Var i) -> i.at) over @ List.filter_map (fun (r : raw) -> List.assoc_opt r.id p.sets) after in
  let holes = holes @ match p.roles with h :: _ -> [ h ] | [] -> [] in
  let home = match holes with [] -> new_hole p | h :: t -> List.fold_left (fun a h -> if h.seq > a.seq then h else a) h t in
  match cached home kind key with
  | Some d -> d
  | None ->
    let d : k reg = match kind with D_r -> Core.r name | D_ur -> Core.ur name | D_r2 -> Core.r2 name in
    fill home (code d);
    home.cache <- (key, Held (kind, d)) :: home.cache;
    p.sets <- (Core.id d, home) :: p.sets;
    d
