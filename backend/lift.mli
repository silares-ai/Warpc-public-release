(** The lifter: warpc's SASS text to PTX, for ptxas to encode -- each
    instruction as the PTX that asks for the same work; the uniform
    datapath's single-issue instructions on lane 0; tensor memory's
    allocation construct as one tcgen05.alloc; the waits and fences PTX asks
    for that have no instruction of their own; each value's live range a
    register of its own. Anything else is refused. The PTX's [// launch:] line
    states its launch: threads, cluster, and dynamic shared memory -- all of
    the program's shared memory, and the allocator's answer past it. *)

(** The PTX of a kernel's SASS text, as [warpc] prints it. *)
val ptx : string list -> string
