let usage () =
  prerr_endline
    "usage: warpc [--sm sm_100a | sm_103a] <command>\n\
    \       warpc gemm M N K                         the program chosen for the shape\n\
    \       warpc one M N K DEPTH                   one CTA a 128 x 128 tile\n\
    \       warpc pair M N K DEPTH                  CUTLASS's orientation, a 2x1 cluster\n\
    \       warpc swapped M N K TM TN DEPTH [--clc] [--ask-ahead] [--direct] [--serpentine]\n\
    \                     [--cluster-n N] [--bufs B] [--group G] [--epi-rows R] [--streamk] [--split S] [--split-grid S] [--split-add] [--late-launch] [--stamps] [--stage-stamps]\n\
    \                                               the swapped orientation: the MMA's A operand is the tile of B^T\n\
    \       warpc expand <any of the above>         the program, one line an instruction\n\
    \       warpc check <any of the above>          every derived operand against its layout, the barriers' protocol and\n\
    \                                               the happens-before order, on the host\n\
    \       warpc hb <any of the above>             the happens-before table: each pair of accesses and what orders it\n\
    \       warpc lift FILE.sass                    the SASS text as PTX, for ptxas";
  exit 2

let int = int_of_string

let build = function
  | [ "gemm"; m; n; k ] -> Programs.choose ~m:(int m) ~n:(int n) ~k:(int k)
  | [ "one"; m; n; k; d ] -> Gemm_one.build ~m:(int m) ~n:(int n) ~k:(int k) ~depth:(int d) ()
  | [ "pair"; m; n; k; d ] -> Gemm_pair.build ~m:(int m) ~n:(int n) ~k:(int k) ~depth:(int d) ()
  | "swapped" :: m :: n :: k :: tm :: tn :: d :: rest ->
    let has f = List.mem f rest in
    let rec value f = function x :: v :: _ when x = f -> Some (int v) | _ :: r -> value f r | [] -> None in
    Gemm_swapped.build ~stamps:(has "--stamps") ~stage_stamps:(has "--stage-stamps") ~streamk:(has "--streamk") ~clc:(has "--clc")
      ~ask_ahead:(has "--ask-ahead") ~late_launch:(has "--late-launch") ~split_add:(has "--split-add") ~direct:(has "--direct") ~serpentine:(has "--serpentine") ?cluster_n:(value "--cluster-n" rest)
      ?bufs:(value "--bufs" rest) ?split:(value "--split" rest) ?split_grid:(value "--split-grid" rest) ?group:(value "--group" rest) ?epi_rows:(value "--epi-rows" rest) ~tile_m:(int tm) ~tile_n:(int tn) ~depth:(int d) ~m:(int m) ~n:(int n) ~k:(int k) ()
  | _ -> usage ()

(* --sm sm_103a: the B300's header and the program chosen for it *)
let sm, args =
  match List.tl (Array.to_list Sys.argv) with
  | "--sm" :: s :: rest when s = "sm_100a" || s = "sm_103a" -> Programs.target := s; s, rest
  | "--sm" :: s :: _ -> failwith ("warpc: no target " ^ s ^ " (sm_100a, sm_103a)")
  | rest -> "sm_100a", rest

let program = build

let () =
  match args with
  | "expand" :: rest -> List.iter print_endline (Compile.expand (program rest))
  | "check" :: rest ->
    let r = Interp.check (program rest) in
    List.iter print_endline (Interp.describe r);
    if r.failures <> [] || r.mismatches > 0 || r.barriers <> [] || r.protocol <> [] || r.memory <> [] || r.races <> [] then exit 1
  | "hb" :: rest ->
    let r = Interp.check ~colored:false (program rest) in
    List.iter print_endline (Interp.table r);
    List.iter print_endline (List.tl (Interp.describe r))
  | [ "lift"; file ] -> print_string (Lift.ptx (In_channel.with_open_text file In_channel.input_lines))
  | args -> List.iter print_endline (Compile.kernel ~sm (program args))
