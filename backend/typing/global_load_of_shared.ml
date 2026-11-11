(* refused: Type Mem.shared = [ `Shared ] is not compatible with type Mem.global = [ `Global ] *)
(* a load from global memory of a shared word *)
let km = Mem.create (Prog.create ())
let words = Mem.words km ~name:"w" ~n:1 ~bytes:4
let () = Mem.ldg_strong km ~desc:(Core.ur2 "desc") (Core.r "x") (Mem.at words [ C 0 ])
