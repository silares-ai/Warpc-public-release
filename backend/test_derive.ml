(* The derivations the compiler makes, checked against what the PTX ISA
   documents and ptxas uses. Every constant below is either the PTX ISA's or
   read out of ptxas-compiled SASS; if a derivation drifts from it, this
   fails before a kernel is ever run. *)

open Layouts

let failures = ref 0

let check name ok =
  if not ok then begin
    incr failures;
    Printf.printf "FAIL %s\n" name
  end

let raises f = match f () with _ -> false | exception Failure _ -> true

let () =
  (* an MMA operand stage: 128 rows of 64 f16, K-major, 128-byte swizzle *)
  let sa = Atom.swizzled_rows ~rows:128 ~cols:64 ~elem:2 in
  let d = Atom.umma_kmajor sa ~elem:2 ~mma_k:16 in
  check "operand SBO is one 8-row group of 128-byte rows" (d.sbo = 1024);
  check "an MMA k step of 16 f16 advances the start by 32 bytes" (d.kstep = 2);
  (* the PTX ISA's shared-memory descriptor, bits 32-63: the stride 1024 >> 4
     at 32-45, the fixed 0b001 at 46-48, 128-byte swizzle (2) at 61-63 *)
  check "descriptor high word is the PTX ISA's for SBO 1024, 128-byte swizzle" (Atom.desc_high d = 0x40004040);
  let bx = Atom.tma_box sa ~elem:2 in
  check "operand box" (bx.box_rows = 128 && bx.box_cols = 64 && bx.box_swizzle = 128);
  (* layouts no descriptor or box can express are refused *)
  let plain = Layout.storage (Group [ Axis { size = 128; stride = 128 }; Axis { size = 64; stride = 2 } ]) in
  check "an unswizzled tile has no SWIZZLE_128B descriptor" (raises (fun () -> Atom.umma_kmajor plain ~elem:2 ~mma_k:16));
  check "an unswizzled tile has no SWIZZLE_128B box" (raises (fun () -> Atom.tma_box plain ~elem:2));
  check "a row that is not one swizzle span is refused" (raises (fun () -> Atom.swizzled_rows ~rows:32 ~cols:48 ~elem:2))

let () =
  (* the accumulator's fragment through tcgen05.ld.32x32b.x32 *)
  let rows = 128 and cols = 256 and n = 32 in
  let bm = rows / 32 and bn = cols / n in
  let ld =
    Layout.compose
      (Layout.interleave ~by:(Linear.canonical (Product [ Bound bm; Bound bn ])) (Atom.ldtm_32x32b ~n))
      (Layout.divide ~by:(Product [ Bound 32; Bound n ]) (Atom.tmem_accumulator ~rows ~cols))
  in
  let at w ch l r = Layout.offset ld (Coord.Tuple [ Tuple [ Idx w; Idx ch ]; Tuple [ Idx l; Idx r ] ]) in
  check "tensor-memory load contract" (not (raises (fun () -> Atom.check_ldtm ~at ~blocks_w:bm ~blocks_ch:bn ~n)));
  let base w ch = Layout.offset ld (Coord.Tuple [ Tuple [ Idx w; Idx ch ]; Tuple [ Idx 0; Idx 0 ] ]) in
  (* warp w of a warpgroup reaches lanes 32 w to 32 w + 31 (PTX ISA,
     tcgen05: the lane in an address's top 16 bits), so its quarter starts
     at + w << 21; chunk ch at column 32 ch *)
  check "block row w starts at tensor-memory lane 32 w" (List.for_all (fun w -> base w 0 = w lsl 21) [ 0; 1; 2; 3 ]);
  check "block column ch starts at column 32 ch" (List.for_all (fun ch -> base 0 ch = 32 * ch) (List.init bn Fun.id))

let () =
  (* the staging write: one warp's 32 x 32 f32 block into the 128-byte
     swizzled box the copy engine reads, lane l's vector q at
     128 l + 16 (q xor (l mod 8)) -- the 128-byte swizzle's closed form *)
  let box = Atom.swizzled_rows ~rows:32 ~cols:32 ~elem:4 in
  let sts = Layout.compose (Atom.ldtm_32x32b ~n:32) box in
  let ok = ref true in
  for l = 0 to 31 do
    for q = 0 to 7 do
      if Layout.offset sts (Coord.Tuple [ Idx l; Idx (4 * q) ]) <> (128 * l) + (16 * (q lxor (l land 7))) then ok := false
    done
  done;
  check "staging addresses match the swizzle's closed form" !ok;
  check "staging write is injective" (Layout.is_injective sts);
  (* and what the emitter will be given evaluates to the same, after its
     range-aware rewrites *)
  let range = function "laneid" -> 32 | v -> failwith v in
  let ok = ref true in
  for q = 0 to 7 do
    let e = Restricted.expr (Restricted.restrict ~at:(Parts [ Free; At (4 * q) ]) sts) in
    let e = Emit.simplify range (Emit.inline [] (Expr.bind "c0" (Expr.var "laneid") e)) in
    for l = 0 to 31 do
      if Expr.eval e (fun _ -> l) <> Layout.offset sts (Coord.Tuple [ Idx l; Idx (4 * q) ]) then ok := false
    done
  done;
  check "emitted staging expressions agree with the layout" !ok;
  let bx = Atom.tma_box box ~elem:4 in
  check "store box" (bx.box_rows = 32 && bx.box_cols = 32 && bx.box_elem = 4 && bx.box_swizzle = 128)

let () =
  (* The transposed accumulator's staging write: register r of lane l of a
     tcgen05.ld.32x32b.x8 is output row r, column l, of an 8 x 32 f32 box
     with the 128-byte swizzle, whose byte offset in closed form is
     (r << 7) + (4 l xor (r << 4)) = 4 l xor (0x90 r). *)
  let box = Atom.swizzled_rows ~rows:8 ~cols:32 ~elem:4 in
  let sts = Layout.compose (Atom.ldtm_32x32b ~n:8) (Layout.compose (Store.transpose ~rows:32 ~cols:8) box) in
  let ok = ref true in
  for l = 0 to 31 do
    for r = 0 to 7 do
      if Layout.offset sts (Coord.Tuple [ Idx l; Idx r ]) <> (4 * l) lxor (0x90 * r) then ok := false
    done
  done;
  check "transposed staging addresses match the swizzle's closed form" !ok

let () =
  (* the emitter's exact division, where a tile count is not a power of two *)
  let ok = ref true in
  List.iter
    (fun (d, bound) ->
      let m, sh = Emit.exact_recip d bound in
      for t = 0 to bound - 1 do
        if (t * m) lsr sh <> t / d then ok := false
      done)
    [ 6, 72; 12, 144; 10, 8192; 3, 4096 ];
  check "exact reciprocal divides every index in range" !ok

let () =
  (* the MMA atom: its instruction descriptor against the format PTX ISA 9.4
     documents (Table 51, .kind::f16: D type F32 = 1 at bits 4-5, A and B
     F16 = 0, no transpose, N >> 3 at bits 17-22, M >> 4 at bits 24-28), and
     the rows of each operand a CTA holds *)
  let atom ~m ~n ~ctas =
    Atom.umma ~ab:F16 ~acc:F32 ~m ~n ~ctas ~a_major:K_major ~b_major:K_major ~a_src:Smem_desc
  in
  let documented ~m ~n = (1 lsl 4) lor ((n lsr 3) lsl 17) lor ((m lsr 4) lsl 24) in
  check "idesc of the 256 x 128 pair MMA is the documented one" (Atom.idesc (atom ~m:256 ~n:128 ~ctas:2) = documented ~m:256 ~n:128);
  check "idesc of the 256 x 256 pair MMA is the documented one" (Atom.idesc (atom ~m:256 ~n:256 ~ctas:2) = documented ~m:256 ~n:256);
  check "idesc of the 128 x 256 MMA is the documented one" (Atom.idesc (atom ~m:128 ~n:256 ~ctas:1) = documented ~m:128 ~n:256);
  check "K of an f16 MMA is 16" (Atom.umma_k (atom ~m:128 ~n:256 ~ctas:1) = 16);
  let u = atom ~m:256 ~n:128 ~ctas:2 in
  let rows l ~cols v = Atom.cta_rows l ~cols ~v in
  check "the pair splits A's rows in halves"
    (rows (Atom.umma_a u) ~cols:16 0 = (0, 128) && rows (Atom.umma_a u) ~cols:16 1 = (128, 128));
  check "the pair splits B's rows in halves"
    (rows (Atom.umma_b u) ~cols:16 0 = (0, 64) && rows (Atom.umma_b u) ~cols:16 1 = (64, 64));
  check "the pair splits the result's rows" (rows (Atom.umma_c u) ~cols:128 1 = (128, 128));
  check "one CTA takes M of 64 or 128 only" (raises (fun () -> atom ~m:256 ~n:128 ~ctas:1));
  check "a pair takes N in steps of 16" (raises (fun () -> atom ~m:256 ~n:136 ~ctas:2));
  check "A from tensor memory is K-major"
    (raises (fun () -> Atom.umma ~ab:F16 ~acc:F32 ~m:128 ~n:128 ~ctas:1 ~a_major:MN_major ~b_major:K_major ~a_src:Tmem))

let () =
  (* every program checks its own derivations as it is built; building and
     compiling them here runs those checks over the shapes the benchmarks
     use: each family, and the program the chooser picks for each shape *)
  let compiles name k = check ("compiles " ^ name) (not (raises (fun () -> Compile.kernel (k ())))) in
  compiles "one CTA a tile, 1024^3" (fun () -> Gemm_one.build ~m:1024 ~n:1024 ~k:1024 ~depth:4 ());
  compiles "the CUTLASS pair, 1536^3" (fun () -> Gemm_pair.build ~m:1536 ~n:1536 ~k:1536 ~depth:8 ());
  compiles "stream-K, 2048^3" (fun () ->
    Gemm_swapped.build ~streamk:true ~direct:true ~tile_m:128 ~tile_n:128 ~depth:8 ~m:2048 ~n:2048 ~k:2048 ());
  List.iter
    (fun (m, n, k) -> compiles (Printf.sprintf "the chosen %dx%dx%d" m n k) (fun () -> Programs.choose ~m ~n ~k))
    [ 1024, 1024, 1024; 1536, 1536, 1536; 2048, 2048, 2048; 4096, 4096, 4096; 4096, 4096, 1024; 3072, 1280, 2048
    ; 8192, 2048, 4096; 8192, 8192, 8192; 16384, 16384, 16384 ];
  if !failures > 0 then (Printf.printf "%d derivation checks failed\n" !failures; exit 1)
  else print_endline "derivations: every check passed"
