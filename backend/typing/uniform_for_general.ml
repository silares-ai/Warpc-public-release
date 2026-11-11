(* refused: Type Core.ur = [ `UR ] is not compatible with type Core.r = [ `R ] *)
(* a uniform register where a general one belongs *)
let _ = Core.iadd3 (Core.r "a") (Core.ur "u") (Core.r "b")
