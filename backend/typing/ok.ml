(* A program the types accept: a wait on a shared barrier at a general
   parity, a load of a shared word into a general register, a stage of a
   ring. *)
let p = Prog.create ()
let km = Mem.create p
let bars = Mem.barriers km ~name:"b" ~n:4 ~count:1
let words = Mem.words km ~name:"w" ~n:4 ~bytes:4
let stage = Prog.ring p ~name:"s" ~depth:4 ~parity:0

let () =
  Prog.step p stage (fun () ->
    Mem.wait km (Mem.at bars [ I stage ]);
    Mem.lds km (Core.r "x") (Mem.at words [ I stage ]))
