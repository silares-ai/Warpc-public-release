(* refused: Type Mem.barrier = [ `Barrier ] is not compatible with type Mem.word = [ `Word ] *)
(* a store of a word to a barrier *)
let km = Mem.create (Prog.create ())
let bars = Mem.barriers km ~name:"b" ~n:1 ~count:1
let () = Mem.sts km (Mem.at bars [ C 0 ]) ~data:(Core.r "x")
