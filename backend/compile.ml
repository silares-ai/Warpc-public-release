(* A core program to SASS text: colour the registers,
   rename every placeholder to its register, schedule, and write the header
   from what the kernel declares. The statements go through one to one. *)

open Core

let round_up a m = (a + m - 1) / m * m

(* the declaration each placeholder unit of the kernel belongs to *)
let owners (k : kernel) =
  let t = Hashtbl.create 256 in
  List.iter (fun (r : raw) -> for i = 0 to r.width - 1 do Hashtbl.replace t (r.id + i) r done) k.regs;
  t

(* the physical register a placeholder unit is coloured to *)
let physical owners colors u =
  if u < first_placeholder then u
  else
    let o = Hashtbl.find owners u in
    match Hashtbl.find_opt colors o.id with
    | Some b -> b + (u - o.id)
    | None -> failwith (Printf.sprintf "%s was never coloured" o.name)

let map_reg f = function
  | Sass.R u -> Sass.R (f u)
  | UR u -> UR (f u)
  | P u -> P (f u)
  | UP u -> UP (f u)

(* A program names only declared registers; the stack pointer R1 is the one
   physical register it may write. *)
let check_physical (ins : Sass.insn) =
  let bad = function
    | Sass.R 1 -> ()
    | (Sass.R u | UR u | P u | UP u) as r when u < first_placeholder ->
      failwith (Printf.sprintf "%s names %s, a register no declaration gave it" (Sass.to_string ins) (Sass.reg_text r))
    | _ -> ()
  in
  List.iter (function Sass.Reg r -> bad r | Text _ | Lab _ -> ()) ins.asm;
  Option.iter (fun (r, _) -> bad r) ins.guard

let rename owners colors (it : Sass.item) : Sass.item =
  match it with
  | Label _ -> it
  | I ins ->
    let reg = map_reg (physical owners colors) in
    (* the effect names the registers it was built with: through the colours *)
    let through (m : Sass.machine) = { m with get = (fun r -> m.get (reg r)); set = (fun r v -> m.set (reg r) v) } in
    let sem = Option.map (fun f m -> f (through m)) ins.sem and act = Option.map (fun f m -> f (through m)) ins.act in
    I
      { ins with
        sem
      ; act
      ; asm = List.map (function Sass.Reg r -> Sass.Reg (reg r) | p -> p) ins.asm
      ; guard = Option.map (fun (r, neg) -> reg r, neg) ins.guard
      ; defs = List.map reg ins.defs
      ; uses = List.map reg ins.uses
      ; late_uses = List.map reg ins.late_uses
      }

(* Labels numbered by the kernel alone: LOOP_1, LOOP_2 in the order they
   are placed, whatever else was built in the same run. *)
let canonical_labels (body : stmt list) =
  let count = Hashtbl.create 16 and names = Hashtbl.create 64 in
  List.iter
    (function
      | Sass.Label l ->
        let prefix = match String.rindex_opt l '_' with Some i -> String.sub l 0 i | None -> l in
        let n = 1 + Option.value (Hashtbl.find_opt count prefix) ~default:0 in
        Hashtbl.replace count prefix n;
        Hashtbl.replace names l (Printf.sprintf "%s_%d" prefix n)
      | I _ -> ())
    body;
  let name l = match Hashtbl.find_opt names l with Some n -> n | None -> failwith ("a branch to " ^ l ^ ", which is never placed") in
  List.map
    (function
      | Sass.Label l -> Sass.Label (name l)
      | I ins ->
        I
          { ins with
            asm = List.map (function Sass.Lab l -> Sass.Lab (name l) | p -> p) ins.asm
          ; branch = (match ins.branch with `Bra l -> `Bra (name l) | b -> b)
          })
    body

let param_line = function
  | Tmap t -> Printf.sprintf ".tmap %s %d %d %d %d %d %d" t.pname t.rows t.cols t.elem t.box_rows t.box_cols t.swizzle
  | Ptr p -> Printf.sprintf ".ptr %s %d %d %d" p.pname p.rows p.cols p.elem

(* Dependent launch: a kernel that takes part -- one with a PREEXIT or an
   ACQBULK -- runs once the grid before it in the stream has passed its
   PREEXIT, so every load or store of global memory -- what the grids before
   may write or read -- comes after an ACQBULK on every path from the entry.
   A tensor map's prefetch and a constant's load read what the host wrote,
   not a grid. A kernel with neither is launched without the attribute. *)
let global_access (ins : Sass.insn) =
  let op = Sass.opcode ins in
  List.exists (fun pre -> String.length op >= String.length pre && String.sub op 0 (String.length pre) = pre)
    [ "LDG"; "STG"; "RED"; "ATOM"; "UTMALDG"; "UTMASTG"; "UTMAREDG" ]

let check_acquire (body : stmt list) =
  let takes_part = List.exists (function Sass.I ins -> List.mem (Sass.opcode ins) [ "PREEXIT"; "ACQBULK" ] | Label _ -> false) body in
  if takes_part then begin
  let arr = Array.of_list body in
  let n = Array.length arr in
  let label_index = Hashtbl.create 16 in
  Array.iteri (fun i it -> match it with Sass.Label l -> Hashtbl.replace label_index l i | I _ -> ()) arr;
  let succ i =
    let fall = if i + 1 < n then [ i + 1 ] else [] in
    match arr.(i) with
    | Sass.Label _ -> fall
    | I ins ->
      (match ins.branch with
       | `No -> fall
       | `Exit -> if ins.guard = None then [] else fall
       | `Bra l -> let t = Hashtbl.find label_index l in if ins.guard = None then [ t ] else t :: fall)
  in
  (* acquired.(i): an ACQBULK has run on every path to item i *)
  let acquired = Array.make n true and seen = Array.make n false in
  let work = Queue.create () in
  acquired.(0) <- false; seen.(0) <- true; Queue.push 0 work;
  while not (Queue.is_empty work) do
    let i = Queue.pop work in
    let out = acquired.(i) || (match arr.(i) with I ins -> Sass.opcode ins = "ACQBULK" | Label _ -> false) in
    List.iter
      (fun j ->
        let v = if seen.(j) then acquired.(j) && out else out in
        if (not seen.(j)) || v <> acquired.(j) then (seen.(j) <- true; acquired.(j) <- v; Queue.push j work))
      (succ i)
  done;
  Array.iteri
    (fun i it ->
      match it with
      | Sass.I ins when seen.(i) && (not acquired.(i)) && global_access ins ->
        failwith (Printf.sprintf "%s touches global memory on a path from the entry with no ACQBULK before it" (Sass.to_string ins))
      | _ -> ())
    arr
  end

(* the program after colouring and renaming, before scheduling, and the
   colouring *)
let colored (k : kernel) =
  List.iter (function Sass.I ins -> check_physical ins | Label _ -> ()) k.body;
  check_acquire k.body;
  (* a label is one place: a statement list used twice needs its own labels *)
  let seen = Hashtbl.create 64 in
  List.iter
    (function
      | Sass.Label l -> if Hashtbl.mem seen l then failwith ("label " ^ l ^ " is placed twice") else Hashtbl.replace seen l ()
      | I _ -> ())
    k.body;
  let owners = owners k in
  let colors = Color.color ~owner:(Hashtbl.find_opt owners) k.body in
  let body = List.map (rename owners colors) (canonical_labels k.body) in
  let max_r =
    List.fold_left
      (fun m it ->
        match it with
        | Sass.I ins -> List.fold_left (fun m r -> match r with Sass.R u -> max m u | _ -> m) m (ins.defs @ ins.uses @ ins.late_uses)
        | Label _ -> m)
      1 body
  in
  body, max_r, (fun u -> physical owners colors u)

let kernel ?(sm = "sm_100a") (k : kernel) : string list =
  let body, max_r, _ = colored k in
  let n_in = List.length (List.filter (function Sass.I _ -> true | _ -> false) k.body) in
  let n_out = List.length (List.filter (function Sass.I _ -> true | _ -> false) body) in
  if n_in <> n_out then failwith "Compile: a statement did not become one instruction";
  let gx, gy = k.grid and tm, tn, tk = k.tile in
  List.map (fun l -> "# " ^ l) k.notes
  @ [ ".kernel " ^ k.kname; ".sm " ^ sm
    (* the two registers above the highest are the hardware's *)
    ; Printf.sprintf ".regs %d" (round_up (max_r + 1 + 2) 8)
    ; Printf.sprintf ".barriers %d" k.named_barriers
    ; Printf.sprintf ".threads %d" k.threads; Printf.sprintf ".smem %d" k.smem
    ; Printf.sprintf ".dynsmem %d" k.dynsmem; Printf.sprintf ".tile %d %d %d" tm tn tk ]
  @ List.map param_line k.params
  @ [ Printf.sprintf ".grid %d %d" gx gy; Printf.sprintf ".mbarriers %d" k.mbarriers; Printf.sprintf ".cluster %d" k.cluster ]
  @ (if k.tcgen05 then [ ".tcgen05" ] else [])
  @ [ ".params " ^ String.concat " " (List.map (fun _ -> "8") k.params @ if k.timing <> No_timing then [ "8" ] else []) ]
  @ (match k.timing with
     | No_timing -> []
     | Stamps l -> ".debug" :: ".stamps" :: List.map (fun (i, s) -> Printf.sprintf "# stamp %d: %s" i s) l
     | Stage_stamps -> [ ".debug"; ".stagestamps" ])
  @ List.map (function Sass.Label l -> l ^ ":" | I ins -> "  " ^ Sass.to_string ins) body

(* The program as written: one line an instruction, each register by the
   name it was declared with. A line the compiler derived from a layout is
   marked +, a line of a construct's control ~; every unmarked line is one of
   the program's own. *)
let expand (k : kernel) : string list =
  let owners = owners k in
  let name r =
    match r with
    | Sass.R u | UR u | P u | UP u when u >= first_placeholder ->
      (match Hashtbl.find_opt owners u with
       | Some o -> if o.width = 1 then o.name else Printf.sprintf "%s.%d" o.name (u - o.id)
       | None -> Sass.reg_text r)
    | r -> Sass.reg_text r
  in
  List.map
    (function
      | Sass.Label l -> l ^ ":"
      | I ins -> (match ins.origin with Program -> "  " | Derived -> "+ " | Construct -> "~ ") ^ Sass.to_string ~name ins)
    (canonical_labels k.body)
