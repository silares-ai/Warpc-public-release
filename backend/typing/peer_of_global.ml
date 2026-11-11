(* refused: Type Mem.global = [ `Global ] is not compatible with type Mem.shared = [ `Shared ] *)
(* another CTA's window on a buffer in global memory *)
let km = Mem.create ~cluster:(2, 1) (Prog.create ())
let flags = Mem.global_words ~name:"f" ~ptr:(Core.r2 "p") ~n:1 ~bytes:4
let _ = Mem.at ~peer:(Mem.peer km ~x:0 ()) flags [ C 0 ]
