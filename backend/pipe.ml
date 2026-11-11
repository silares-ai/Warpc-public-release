(* Pipes between warps: [depth] stages, each with a barrier pair -- full once
   its producers have filled it, empty once its consumers have freed it --
   and a ring position at each end. The two ends are different types, so a
   producer can only wait for a stage to be empty and fill it, a consumer
   only wait for it to be full and free it; each end's ring starts at the
   phase its first wait passes at, and flips it as it wraps. Every operation
   is one instruction, or a spin on one. *)

type t =
  { name : string
  ; depth : int
  ; full : (Mem.shared, Mem.barrier) Mem.buffer (* [fills] arrivals fill a stage *)
  ; empty : (Mem.shared, Mem.barrier) Mem.buffer (* [frees] arrivals free it *)
  }

type ring = (Prog.ring, Core.ur) Prog.index

type producer =
  { pk : Mem.kernel
  ; pp : t
  ; pring : ring
  }

type consumer =
  { ck : Mem.kernel
  ; cp : t
  ; cring : ring
  }

(* a pipe of [depth] stages, a stage filled by [fills] arrivals and freed by
   [frees] *)
let make k ~name ~depth ~fills ~frees =
  let full = Mem.barriers k ~name:(name ^ "_full") ~n:depth ~count:fills in
  let empty = Mem.barriers k ~name:(name ^ "_empty") ~n:depth ~count:frees in
  { name; depth; full; empty }

(* this warp's end: every stage free to the producer, none full to the
   consumer *)
let producer k t = { pk = k; pp = t; pring = Prog.ring (Mem.prog k) ~name:t.name ~depth:t.depth ~parity:Barrier.free_parity }
let consumer k t = { ck = k; cp = t; cring = Prog.ring (Mem.prog k) ~name:t.name ~depth:t.depth ~parity:Barrier.full_parity }

(* one stage at this end: [body] runs at it, and the end moves on after *)
let produce e body = Prog.step (Mem.prog e.pk) e.pring body
let consume e body = Prog.step (Mem.prog e.ck) e.cring body

(* [buf]'s copy for the end's stage, in this CTA *)
let slot_p e buf = Mem.at buf [ I e.pring ]
let slot_c e buf = Mem.at buf [ I e.cring ]

(* the producer's side: until the stage is empty, and the barrier the
   stage's fill reports to -- this CTA's, or at [peer] another's *)
let wait_empty e = Mem.wait e.pk (Mem.at e.pp.empty [ I e.pring ])
let full ?peer e = Mem.at ?peer e.pp.full [ I e.pring ]

(* the consumer's side: until the stage is full, a question about it whose
   answer lands in [into] later, and the barrier its release reports to *)
let wait_full e = Mem.wait e.ck (Mem.at e.cp.full [ I e.cring ])
let probe_full e ~into = Mem.probe e.ck (Mem.at e.cp.full [ I e.cring ]) ~into
let empty ?peer e = Mem.at ?peer e.cp.empty [ I e.cring ]
