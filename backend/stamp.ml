(* A timing build: where a warp passes a point the program marks, it writes
   the global timer's low word to the stamp buffer -- the kernel's last
   parameter, 64 words a CTA: 16 events for each of its first four tiles,
   the tile counted mod 4 (run_tma.py's .stamps). A stamp is a construct:
   its instructions are the timing build's, not the program's. *)

open Core

let at p ~nparams ~ev ?count () =
  let t = r "stamp_time" and a = r2 "stamp_addr" and c = r "stamp_cta" and d = ur2 "stamp_desc" in
  Prog.constructs p
    ([ s2r_timer t; ldc64 a (param_off nparams); ldcu64 d gmem_desc_off; s2r_ctaid c ~axis:"X"; imad_wide a c 256 a ]
    @ (match count with
       | Some n -> let q = r "stamp_tile" in [ lop3_and q n 3; imad_wide a q 64 a ]
       | None -> [])
    @ [ stg32 ~desc:d ~base:a ~imm:(4 * ev) ~data:t ])

(* A stamp each time a stage lands: 4096 words a CTA, the [count]-th at
   word count mod 4096 (run_tma.py's .stagestamps) *)
let stage p ?(half = 0) ~nparams ~count () =
  let t = r "stamp_time" and a = r2 "stamp_addr" and c = r "stamp_cta" and d = ur2 "stamp_desc" and q = r "stamp_n" in
  Prog.constructs p
    [ s2r_timer t; ldc64 a (param_off nparams); ldcu64 d gmem_desc_off; s2r_ctaid c ~axis:"X"; imad_wide a c (4 * 4096) a
    ; lop3_and q count 2047; imad_wide a q 8 a; stg32 ~desc:d ~base:a ~imm:(4 * half) ~data:t ]
