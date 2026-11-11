(* refused: Type Core.ur = [ `UR ] is not compatible with type Core.r = [ `R ] *)
(* a general value asked into a uniform register *)
let km = Mem.create (Prog.create ())
let _ = Mem.get km ~cls:General ~into:(Core.ur "u") (Mem.const 0)
