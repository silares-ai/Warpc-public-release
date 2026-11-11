(** The host-side interpreter: a kernel's instructions run for every warp of
   a cluster's CTAs together -- the first cluster and the last -- with each
   instruction's effect on the registers ({!Sass.sem}) and on what orders
   the warps ({!Sass.act}). The warps take turns, each running until it
   waits for something not yet there.

   What the hardware does outside the registers is modelled as the
   protocols need it: an mbarrier counts its phase's arrivals and bytes, a
   wait passes once the phase of its parity has completed; a named barrier
   releases once its count of threads has arrived, the cluster's barrier
   once every warp has; the copy engine's loads land and the tensor core's
   commits arrive as they are issued; tensor memory is handed out in order;
   a flag in global memory reads as raised; the scheduler has no work to
   hand out.

   Every operand the compiler derived is checked where its instruction runs
   against what the layouts it came from say it must be ({!Core.check}),
   and again in the coloured program. Every named barrier's count is
   checked against the warps that arrive at it.

   The barriers' protocol: every warp leaves -- one still waiting is a hang,
   reported with what it waits for and where the barrier stands -- every
   mbarrier ends at the start of a phase, nothing arrives where no barrier
   was initialised, no warp waits on another CTA's barrier.

   The happens-before order the run's synchronisation makes, by vector
   clocks over the warps, the tensor core's queue of each warp's MMAs, and
   each copy-engine operation: two accesses to a place in shared or tensor
   memory, one a write, must be ordered; a generic-proxy write must be
   fenced before the copy engine or the tensor core reads it; what touches a
   CTA's memory from outside its warps must be ordered before they have all
   exited. Each ordered pair is a row of the happens-before table. *)

type failure = {
  what : string;
  at : string;
  cta : int;
  warp : int;
  lane : int;
  got : int;
  want : int;
}

(** Two accesses to a buffer, at least one a write, the earlier ordered
    before the later by [via]; [count] such pairs in the run. *)
type row = { buffer : string; first : string; second : string; via : string; count : int }

type report = {
  checked : int;
  failures : failure list;
  mismatches : int;
  barriers : string list;
  protocol : string list;
  memory : string list;  (** parameters and memory used past what was declared; accumulators added to or read out of turn *)
  races : string list;
  table : row list;
  idle : string list;  (** synchronisation the run passed that ordered no pair *)
}

(** Every derived operand of [k] against its reference, in the program as
   written and as coloured; the barriers' protocol and the happens-before
   order in the program as written. *)
val check : ?colored:bool -> Core.kernel -> report

val describe : report -> string list

(** The happens-before table, a line a row, and the synchronisation that
    ordered nothing. *)
val table : report -> string list

(** Which warps of which CTA of the first cluster run an instruction in the
    host run, by its place among the kernel's instructions: (rank, warp)
    pairs, none for one the run never reaches. *)
val executions : Core.kernel -> int -> (int * int) list
