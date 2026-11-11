(* refused: Type Core.r = [ `R ] is not compatible with type Core.r2 = [ `R2 ] *)
(* one register where a wide operand takes a pair *)
let _ = Core.imad_wide (Core.r "d") (Core.r "a") 4 (Core.r2 "b")
