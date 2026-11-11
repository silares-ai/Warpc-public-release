(* refused: Type Prog.counter = [ `Counter ] is not compatible with type Prog.ring = [ `Ring ] *)
(* a counter stepped as a ring *)
let p = Prog.create ()
let () = Prog.for_ p ~name:"k" ~count:(Times 2) (fun kk -> Prog.step p kk ignore)
