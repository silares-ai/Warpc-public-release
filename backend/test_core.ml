(* The core language's contract: a statement is one instruction, registers
   are coloured without clobbering what is live, wide operands are aligned,
   and a program that does not fit or reads what it never wrote is refused. *)

open Core

let failures = ref 0

let check name ok =
  if not ok then begin
    incr failures;
    Printf.printf "FAIL %s\n" name
  end

let fails_with sub f =
  match f () with
  | _ -> false
  | exception Failure msg ->
    let n = String.length sub and m = String.length msg in
    let rec at i = i + n <= m && (String.sub msg i n = sub || at (i + 1)) in
    at 0

let kernel body =
  { kname = "t"; params = []; threads = 32; grid = 1, 1; cluster = 1; together = 1; smem = 0x400; dynsmem = 0; mbarriers = 1
  ; named_barriers = 1; tile = 1, 1, 1; tcgen05 = false; body; regs = regs_of body; checks = []; touches = []; regions = []; notes = []; timing = No_timing }

(* the physical register a declared one was given *)
let colour body x = Hashtbl.find (Color.color body) (id x)

let () =
  (* one statement, one instruction: the GEMM's instruction lines are its
     statements *)
  let k = Gemm_one.build ~m:256 ~n:256 ~k:256 ~depth:4 () in
  let lines = Compile.kernel k in
  let insns = List.filter (fun l -> String.length l > 2 && String.sub l 0 2 = "  ") lines in
  let stmts = List.filter (function Sass.I _ -> true | Label _ -> false) k.body in
  check "every statement is one instruction" (List.length insns = List.length stmts);
  check "the expansion prints one line a statement" (List.length (Compile.expand k) = List.length k.body);

  (* two values live at once get different registers; one that dies frees its register *)
  let a = r "a" and b = r "b" and c = r "c" and d = r "d" in
  let body = [ mov_imm a 1; mov_imm b 2; iadd3 c a b; mov_imm d 3; iadd3 d d c; exit () ] in
  let col = Color.color body in
  let at x = Hashtbl.find col (id x) in
  check "values live together are apart" (at a <> at b);
  check "a result may take a dead operand's register" (at c = at a || at c = at b || at c = 0);
  check "d is apart from c while both are live" (at d <> at c);
  check "R1 is never handed out" (List.for_all (fun x -> at x <> 1) [ a; b; c; d ]);

  (* pairs start on an even register, quads on a multiple of four *)
  let x = r "x" and pr = r2 "pair" and q = ur4 "quad" and u = ur "u" in
  let never = r "never_written" in
  check "a register read before it is written is refused"
    (fails_with "read before" (fun () -> Color.color [ mov_imm x 1; iadd3 x x never; exit () ]));
  let m = ur2 "m" in
  let body =
    [ mov_imm x 1; mov_rz (part pr 0); mov_rz (part pr 1); imad_wide pr x 4 pr; umov u 0; umov (upart q 0) 0; umov (upart q 1) 0
    ; umov (upart q 2) 0; umov (upart q 3) 0; uiadd3 u u 1; ldcu64 m 0x380; utmaldg ~g:q ~map:m (); stg32 ~desc:m ~base:pr ~imm:0 ~data:x; exit () ]
  in
  check "a pair starts on an even register" (colour body pr mod 2 = 0);
  check "a quad starts on a multiple of four" (colour body q mod 4 = 0);
  check "a uniform pair is even" (colour body m mod 2 = 0);

  (* more values live at once than the class has registers: an error naming the point *)
  let many = List.init 260 (fun i -> r (Printf.sprintf "v%d" i)) in
  let sum = r "sum" in
  let body =
    List.map (fun v -> mov_imm v 1) many @ [ mov_rz sum ] @ List.map (fun v -> iadd3 sum sum v) many @ [ exit () ]
  in
  check "a program that does not fit is refused, not spilled" (fails_with "does not fit" (fun () -> Color.color body));
  check "the error names the pressure" (fails_with "The most live at once is 261" (fun () -> Color.color body));

  (* dependent launch: a kernel with a PREEXIT or an ACQBULK touches global
     memory only after an ACQBULK on every path from the entry *)
  let desc = ur2 "desc" and base = r2 "base" and v = r "v" and q = p "q" in
  let store = stg32 ~desc ~base ~imm:0 ~data:v in
  let acquire body = fails_with "no ACQBULK before it" (fun () -> Compile.colored (kernel body)) in
  check "a store before ACQBULK is refused" (acquire [ preexit (); store; acqbulk (); exit () ]);
  check "a store ACQBULK reaches on one path only is refused"
    (acquire [ preexit (); bra q "SKIP"; acqbulk (); Sass.Label "SKIP"; store; exit () ]);
  check "a store after ACQBULK on every path is not" (not (acquire [ preexit (); acqbulk (); bra q "SKIP"; Sass.Label "SKIP"; store; exit () ]));
  check "a kernel with neither is not checked" (not (acquire [ store; exit () ]));

  (* no physical register but the stack pointer *)
  let raw = one (fun b -> Sass.mov_imm b 5 0) in
  check "a physical register is refused" (fails_with "no declaration gave it" (fun () -> Compile.kernel (kernel [ raw; exit () ])));

  (* ---- memory operands: addresses derived from layouts, and what is refused ---- *)
  let has sub (text : string) =
    let n = String.length sub and m = String.length text in
    let rec at i = i + n <= m && (String.sub text i n = sub || at (i + 1)) in
    at 0
  in
  let texts body = List.filter_map (function Sass.I i -> Some (Sass.to_string i) | Label _ -> None) body in
  let setup () =
    let p = Prog.create () in
    let km = Mem.create p in
    let l = Atom.swizzled_rows ~rows:128 ~cols:64 ~elem:2 in
    p, km, Mem.ring km ~name:"sa" ~depth:4 ~elem:2 l
  in
  (* a stage's address is one multiply-add from the stage, where the
     iteration asks about it; a counter's value an add where the loop steps *)
  let p, km, sa = setup () in
  let st = Prog.ring p ~name:"stage" ~depth:4 ~parity:0 in
  Prog.for_ p ~name:"k" ~count:(Times 8) (fun kk ->
    Prog.step p st (fun () ->
      let a = Mem.addr km ~cls:Uniform (Mem.at sa [ I st ]) and x = Mem.get km ~cls:Uniform Mem.(scale 64 (idx kk)) in
      Prog.emit p (uiadd3_uu a a x)));
  let body = Prog.finish p in
  check "a stage's address is one multiply-add by the copy's bytes" (List.length (List.filter (fun t -> has "UIMAD" t && has "0x4000" t) (texts body)) = 1);
  check "a counter's value steps by its stride" (List.exists (fun t -> has "UIADD3" t && has "0x40," t) (texts body));
  check "derived lines are marked in the expansion"
    (List.exists (fun l -> String.length l > 2 && String.sub l 0 2 = "+ ") (Compile.expand (kernel (body @ [ exit () ]))));
  (* the refusals *)
  let p, km, sa = setup () in
  check "a coordinate outside its mode is refused" (fails_with "a coordinate out of range" (fun () -> Mem.addr km ~cls:Uniform (Mem.at sa [ C 4 ])));
  check "a term names every index mode" (fails_with "one coordinate an index mode" (fun () -> Mem.addr km ~cls:Uniform (Mem.at sa [])));
  let deep = Prog.ring p ~name:"deep" ~depth:8 ~parity:0 in
  check "an index that ranges past its mode is refused" (fails_with "ranges past the mode" (fun () -> Mem.addr km ~cls:Uniform (Mem.at sa [ I deep ])));
  check "a step inside a step of the same ring is refused" (fails_with "asked twice" (fun () -> Prog.step p deep (fun () -> Prog.step p deep ignore)));
  let other = Prog.ring p ~name:"other" ~depth:2 ~parity:0 in
  let u = ur "into" in
  ignore (Mem.addr km ~cls:Uniform ~into:u (Mem.at sa [ C 0 ]));
  check "a register given two values is refused" (fails_with "already holds" (fun () -> Mem.addr km ~cls:Uniform ~into:u (Mem.at sa [ C 1 ])));
  check "a fixed window offset is refused an index" (fails_with "a window offset is fixed" (fun () -> Mem.window_imm (Mem.at sa [ I other ])));
  Prog.for_ p ~name:"i" ~count:(Times 4) (fun i ->
    Prog.for_ p ~name:"j" ~count:(Times 4) (fun j ->
      check "a value moving with two counters is refused"
        (fails_with "moves with more than one index" (fun () -> Mem.get km ~cls:Uniform Mem.(plus (idx i) (idx j))));
      check "a value not affine in its counter is refused"
        (fails_with "is not affine in" (fun () -> Mem.get km ~cls:Uniform (Mem.of_expr (Layouts.Expr.div (Layouts.Expr.var "j") 2) [ "j", Prog.Var j ])))));
  let p2 = Prog.create () in
  let km2 = Mem.create p2 in
  check "a ring past shared memory is refused"
    (fails_with "shared memory" (fun () -> Mem.ring km2 ~name:"big" ~depth:16 ~elem:2 (Atom.swizzled_rows ~rows:256 ~cols:64 ~elem:2)));

  (* ---- the host-side check of every derived operand ---- *)
  let k = Gemm_one.build ~m:256 ~n:256 ~k:512 ~depth:4 () in
  let r = Interp.check k in
  check "every derived operand is what its layout says" (r.failures = [] && r.checked > 0);
  check "the colouring keeps every derived operand" (r.mismatches = 0);
  (* a derived address off by 16 bytes, and one never computed: both caught *)
  let tamper f =
    let first = ref true in
    { k with
      body =
        List.concat_map
          (function
            | Sass.I ins when !first && ins.origin = Sass.Derived && String.length (Sass.to_string ins) > 5 && String.sub (Sass.to_string ins) 0 5 = "UIMAD" ->
              first := false;
              f ins
            | it -> [ it ])
          k.body }
  in
  let off_by_16 (ins : Sass.insn) =
    let d = List.hd ins.defs in
    [ Sass.I { ins with sem = Some (fun m -> Option.get ins.sem m; m.set d (m.get d + 16)) } ]
  in
  check "a derived address that is wrong is caught" ((Interp.check (tamper off_by_16)).failures <> []);
  check "a derived instruction that is missing is caught" ((Interp.check ~colored:false (tamper (fun _ -> []))).failures <> []);
  check "and refused by the colouring" (fails_with "read before they are written" (fun () -> Compile.kernel (tamper (fun _ -> []))));

  (* ---- a named barrier's count against the warps that arrive at it ---- *)
  check "the program's named barriers are made by the warps that arrive" ((Interp.check ~colored:false k).barriers = []);
  let bar1 f = { k with body = List.concat_map (function Sass.I ins when ins.named_barrier = Some 1 -> f ins | it -> [ it ]) k.body } in
  let count c (ins : Sass.insn) = [ Sass.I { ins with asm = Sass.pieces (Printf.sprintf "BAR.SYNC.DEFER_BLOCKING 0x1, 0x%x" c) } ] in
  check "a named barrier's count too low is caught" ((Interp.check ~colored:false (bar1 (count 0x80))).barriers <> []);
  let dropped = ref false in
  let drop_one (ins : Sass.insn) = if !dropped then [ Sass.I ins ] else (dropped := true; []) in
  check "a warp's missing arrival is caught" ((Interp.check ~colored:false (bar1 drop_one)).barriers <> []);

  (* ---- the barriers' protocol and the happens-before order: the
     program keeps them, and each way of breaking them is caught ---- *)
  let run k = Interp.check ~colored:false k in
  let r = run k in
  check "the program's barriers keep their protocol" (r.protocol = []);
  check "every pair of the program's accesses is ordered" (r.races = []);
  check "the happens-before table orders a stage's load before its MMAs by the full barrier"
    (List.exists (fun (row : Interp.row) -> row.buffer = "sa" && has "UTMALDG" row.first && has "UTCHMMA" row.second && row.via = "wait on stage_full") r.table);
  let op (i : Sass.insn) = Sass.opcode i in
  let edit (k : kernel) f = { k with body = f k.body } in
  let first_only p f body =
    let done_ = ref false in
    List.concat_map (function Sass.I i when (not !done_) && p i -> done_ := true; f i | it -> [ it ]) body
  in
  let hangs k = List.exists (has "hang:") (run k).protocol in
  check "a dropped arrival is a hang, and is caught" (hangs (edit k (first_only (fun i -> op i = "SYNCS.ARRIVE.TRANS64") (fun _ -> []))));
  (* the expected bytes 16 more than the loads bring *)
  let tx_plus_16 (i : Sass.insn) = let d = List.hd i.defs in [ Sass.I { i with sem = Some (fun m -> Option.get i.sem m; m.set d (m.get d + 16)) } ] in
  check "a stage told the wrong byte count hangs, and is caught" (hangs (edit k (first_only (fun i -> op i = "MOV" && has ", 0x8000" (Sass.to_string i)) tx_plus_16)));
  check "a missing commit is a hang, and is caught" (hangs (edit k (first_only (fun i -> op i = "UTCBAR") (fun _ -> []))));
  (* the stage released before its MMAs have read it: the next load into it races them *)
  let early body =
    let commit = List.find (function Sass.I i -> op i = "UTCBAR" | _ -> false) body in
    let body = first_only (fun i -> op i = "UTCBAR") (fun _ -> []) body in
    first_only (fun i -> op i = "UTCHMMA") (fun i -> [ commit; Sass.I i ]) body
  in
  check "a stage freed before its MMAs have read it is caught" (List.exists (has "nothing orders") (run (edit k early)).races);
  (* the staged store without its proxy fence *)
  let staged = Gemm_swapped.build ~tile_m:128 ~tile_n:128 ~depth:4 ~m:256 ~n:256 ~k:256 () in
  check "a two-CTA program keeps its barriers' protocol" ((run staged).protocol = []);
  check "a staging copy the copy engine reads unfenced is caught"
    (List.exists (has "no proxy fence") (run (edit staged (List.filter (function Sass.I i -> op i <> "FENCE.VIEW.ASYNC.S" | _ -> true)))).races);

  (* ---- memory as declared: parameters, global memory, the accumulator ---- *)
  let mem k = (run k).memory in
  check "the program uses memory as it declares it" (mem k = []);
  let small_c = { k with params = List.map (function Ptr p -> Ptr { p with rows = p.rows / 2 } | x -> x) k.params } in
  check "a store past its parameter's end is caught" (List.exists (has "past its") (mem small_c));
  let swapped_params = { k with params = (match k.params with c :: a :: rest -> a :: c :: rest | ps -> ps) } in
  check "a pointer taken as a tensor map is caught" (List.exists (has "a pointer, as its tensor map") (mem swapped_params));
  let map_first = { k with params = (match k.params with _ :: a :: rest -> a :: a :: rest | ps -> ps) } in
  check "a tensor map stored through is caught" (List.exists (has "a tensor map, as memory") (mem map_first));
  let nth_mma n f body =
    let seen = ref 0 in
    List.map (function Sass.I i when op i = "UTCHMMA" -> incr seen; if !seen = n then Sass.I (f i) else Sass.I i | it -> it) body
  in
  let accumulating (i : Sass.insn) = { i with act = Some (fun _ -> Sass.Mma { accumulate = true }) }
  and overwriting (i : Sass.insn) = { i with act = Some (fun _ -> Sass.Mma { accumulate = false }) } in
  check "a tile's first MMA adding to a fresh accumulator is caught" (List.exists (has "since its allocation") (mem (edit k (nth_mma 1 accumulating))));
  check "an MMA overwriting results no load has read is caught" (List.exists (has "no load has read") (mem (edit k (nth_mma 2 overwriting))));
  (* stream-K's hand-off through global memory: the partial, its fence, its flag *)
  let sk = Gemm_swapped.build ~streamk:true ~direct:true ~tile_m:128 ~tile_n:128 ~depth:8 ~m:2048 ~n:2048 ~k:512 () in
  let r = run sk in
  check "stream-K's hand-off keeps its protocol and its order" (r.protocol = [] && r.races = [] && r.memory = []);
  check "the partial is ordered by its flag"
    (List.exists (fun (row : Interp.row) -> row.buffer = "ws" && row.via = "strong load of flags") r.table);
  check "a partial written without MEMBAR.ALL.GPU is caught"
    (List.exists (has "no MEMBAR.ALL.GPU") (run (edit sk (List.filter (function Sass.I i -> op i <> "MEMBAR.ALL.GPU" | _ -> true)))).races);
  (* the adding CTA's other warps reading the partial without the barrier
     that follows its first warp's wait for the flag *)
  let second_bar2 body =
    let seen = ref 0 in
    List.filter (function Sass.I i when i.named_barrier = Some 2 -> incr seen; !seen <> 2 | _ -> true) body
  in
  check "a partial read without the barrier after the flag is caught" (List.exists (has "nothing orders") (run (edit sk second_bar2)).races);

  (* ---- the expansion: every unmarked line is one of the program's ---- *)
  let lines = Compile.expand k in
  let loop =
    (* the MMA warp's k loop: from its label to its branch back *)
    let rec from = function [] -> [] | l :: rest -> if String.length l > 4 && String.sub l 0 4 = "LOOP" && List.exists (fun x -> has "UTCHMMA" x) (take_loop rest) then take_loop rest else from rest
    and take_loop = function [] -> [] | l :: rest -> if has "BRA LOOP" l then [ l ] else l :: take_loop rest in
    from lines
  in
  let own = List.filter (fun l -> String.length l > 2 && String.sub l 0 2 = "  ") loop in
  let op l = List.hd (String.split_on_char ' ' (String.trim l)) in
  check "the MMA loop's unmarked lines are the program's"
    (List.map op own = [ "UISETP.NE.U32.AND"; "SYNCS.PHASECHK.TRANS64.TRYWAIT"; "UTCHMMA"; "UTCHMMA"; "UTCHMMA"; "UTCHMMA"; "UMOV"; "UTCBAR" ]);
  check "a loop's control is marked as the construct's" (List.exists (fun l -> String.sub l 0 2 = "~ " && has "BRA LOOP" l) loop);
  check "a stage's addresses are marked as derived" (List.exists (fun l -> String.sub l 0 2 = "+ " && has "UIMAD" l) loop);

  (* ---- a kernel is a value: built twice, it compiles to the same text ---- *)
  let again = Gemm_one.build ~m:256 ~n:256 ~k:512 ~depth:4 () in
  check "a kernel built twice compiles to the same SASS" (Compile.kernel k = Compile.kernel again);
  check "and expands to the same program" (Compile.expand k = Compile.expand again);

  if !failures > 0 then (Printf.printf "%d core checks failed\n" !failures; Stdlib.exit 1) else print_endline "core: every check passed"
