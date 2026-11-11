(** A core program to SASS text: colour the registers,
   rename every placeholder to its register, and write the header from what
   the kernel declares. The statements go through one to one, in program
   order: the text is what [warpc lift] reads, and ptxas schedules it. *)

(** the program after colouring and renaming, and the colouring *)
val colored : Core.kernel -> Sass.item list * int * (int -> int)

(** [sm] the target the header names: sm_100a (the default, the B200) or
    sm_103a (the B300). *)
val kernel : ?sm:string -> Core.kernel -> string list

(** The program as written: one line an instruction, each register by the
   name it was declared with. A line the compiler derived from a layout is
   marked +, a line of a construct's control ~; every unmarked line is one of
   the program's own. *)
val expand : Core.kernel -> string list
