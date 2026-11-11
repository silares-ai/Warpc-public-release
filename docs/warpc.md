# warpc reference

The full description of the language, its checks and the lifter. The
[README](../README.md) is the overview.

## The language

A program is a list of statements, and a statement is one SASS instruction
or a label. Its operands are registers the program declares by name -- `r`,
`ur`, `p`, `up`, pairs and quads aligned -- and `backend/color.ml` places
them: liveness over the program's control flow, an interference graph per
register class, a greedy colouring; when they do not fit it stops and names
the point of greatest pressure and what is live there. The SASS warpc
writes never spills; ptxas allocates the registers again from the PTX.
The loops, the ring of stages, the protocols between warps and the order of
every instruction are the program's text; ptxas schedules the PTX it is
lifted to.

Registers, buffers and indices carry their kinds in OCaml's types. A
register's class and width are its type -- `r reg`, `r2 reg`, `ur4 reg`,
`p reg`, `up reg`, a vector `rv reg` -- and every instruction states the
registers it takes, so a uniform register where a
general one belongs, one register where a pair belongs, or a predicate
where a value belongs does not compile. A buffer's space and kind are its
type -- `(shared, barrier)`, `(shared, tile)`, `(shared, word)`, `(global,
word)`, `(global, slab)` -- so a wait takes a shared barrier, an MMA shared
tiles, a load past the L1 a global word, and another CTA's window is a
window on shared memory only. An index's kind is its type -- fixed, a
counter, a ring position -- and only a ring position is stepped. The
instructions under the language (`backend/sass.ml`) are untyped data; the
types erase and add no instruction. `backend/typing/` holds programs the types
refuse, each required to fail to compile with its refusal. What a type does
not carry is checked where the program is built: a vector's length, a
coordinate's range, a register asked to hold two values.

An instruction is data (`backend/sass.ml`): its text as literal pieces and
register operands, its guard, the registers it writes, reads and reads
late (after issue), and its effect on the registers. A few lines are not one of the
program's own, and each is marked where it lands: the control of the
language's constructs -- a loop's count, compare and branch, a ring's step,
a role's dispatch, a spin's sleep and branch back, the barriers'
initialisation, the tensor-memory allocation -- and the address arithmetic
the compiler derives.

## Addresses

A shared- or global-memory address is not the program's text. A load, a
store, a barrier operation, a copy-engine transfer or an MMA takes its
shared- or global-memory operand as a buffer at coordinates, and the
compiler forms the address from the buffer's layout. An accumulator in
tensor memory is the allocation's base, plus the program's offset for the
second of two accumulators, and the columns inside it are derived from its
layout:

- A buffer is declared with its layout (`backend/mem.ml`): a ring of
  operand tiles, a row of staging copies a warp, an array of barriers, the
  words tensor-memory addresses are left in, a workspace in global memory.
  Its index modes -- the ring's stage, a warp's copy, an accumulator buffer,
  a CTA's slot -- take coordinates; its element layout is the tile the
  instruction fixes. `Mem` places shared memory and refuses what does not
  fit.
- Warps meet through pipes (`backend/pipe.ml`): stages with a full and an
  empty barrier each, and a ring position at each end. The producer's end
  and the consumer's end are different types -- a producer can only wait
  for a stage to be empty and fill it, a consumer only wait for it to be
  full and free it -- and each end's ring starts at the phase its first
  wait passes at.
- A coordinate is a compile-time integer, an index, or a value of indices
  (`backend/prog.ml`). An index is a register with a range: fixed where it
  is set (the warp, the lane, the tile, the CTA's rank in its cluster), a
  counter, or a ring position that wraps and flips its phase's parity.
- The address of an operand is the buffer's layout at the coordinates,
  composed with the instruction's fragment where the lanes differ. Its
  strided form over the indices is derived into holes the program leaves as
  it is built: a value of fixed indices once, where the last of them is set;
  a value affine in a counter as an induction, set where the loop starts
  and stepped where it steps; a value of a ring position as one
  multiply-add from the stage at the top of each iteration, ahead of the
  wait. A constant part becomes the instruction's immediate.
- An operand the buffer's layout does not allow is refused: a coordinate
  outside its mode, an index whose range passes the mode, a register asked
  to hold two values, a value moving with two indices, a descriptor that
  does not start on a swizzle row.

## Checks

Every derived operand carries a check: what it must hold where its
instruction runs, computed from the buffer's layout at the coordinates --
not from the expression the derivation compiled. `warpc check` runs the
kernel's instructions on the host (`backend/interp.ml`) for every warp of
the first and last clusters, lanes together as a warp runs them, and
compares every derived operand with its reference; it runs the coloured
program too and requires each checked operand to hold the same value
there. It checks every named barrier as well: the k-th arrivals of the
warps that reach a barrier must all state one thread count, and that count
must be 32 times how many warps they are -- a count too high hangs, one too
low lets warps through early -- and a warp must arrive with all its lanes.

The warps of a cluster's CTAs run together, each until it waits for
something not yet there. Each instruction that orders the warps says what
it does with its operands' values (`Sass.act`): a barrier's
initialisation, a lane's arrival and the bytes it states, a wait and its
parity, a copy-engine load or store, an MMA, a commit, a proxy fence, the
cluster's barrier, a request to the scheduler. An mbarrier counts its
phase's arrivals and bytes and completes the phase when both reach what it
takes; a wait passes once the phase of its parity has completed, as the
hardware's try-wait answers. The copy engine's bytes and the tensor core's
commits arrive as they are issued -- the earliest the hardware could. A
multicast load's bytes report at the barrier's offset in each CTA they land
in, the pair's first CTA for a load that feeds a two-CTA MMA. The protocol
check: every warp must leave -- a warp still waiting, or spinning on a word
of global memory nothing in the run changes, is a hang, reported with what
it waits for and where the barrier stands -- every mbarrier must end at the
start of a phase, nothing may arrive where no barrier was initialised, and
no warp may wait on another CTA's barrier. A check runs the first cluster
and the last; a kernel that reads global memory another CTA writes (stream-K's
partials and flags) runs its first two clusters together, so the hand-off
between them runs too. A global word no CTA of the run writes reads as 1, a
flag raised by a CTA outside the run.

Memory must be used as declared: a constant the kernel reads is a
parameter it declares; a global access lands inside the pointer parameter
it addresses (a timing build's stamp buffer is declared by its timing); a
copy-engine operation takes a tensor-map parameter and its box fits the
shared copy it fills or empties. Tensor memory keeps each column's state:
an MMA that adds to an accumulator adds to one an MMA has written since it
was allocated or last read out -- a tile's first MMA overwrites -- one that
overwrites does not overwrite results no load has read, and a load reads
what an MMA wrote.

The run also keeps a vector clock for each thread of the happens-before
order: each warp, the tensor core's queue of a warp's MMAs and commits
(they run in order), and each copy-engine load or store and each request to
the scheduler (they run on their own). A barrier's phase hands the clocks
of what arrived to whoever waits for it; a named barrier, the cluster's
barrier, a wait for the copy engine's stores and a strong load of what a
strong store wrote do the same. Every access to shared and global memory (4
bytes at a time), tensor memory (a column of a lane quarter) and the
barriers' words is checked against the accesses before it: two to the same
place, one a write, must be ordered -- two strong accesses are the
synchronisation itself, and the stamp buffer's words are written by several
warps by design; a write through the generic proxy must be fenced
(`FENCE.VIEW.ASYNC`) before the copy engine or the tensor core reads it; a
global write another CTA reads must be fenced at the GPU's scope
(`MEMBAR.ALL.GPU`) before what orders the read; whatever touches a CTA's
memory from outside its warps must be ordered before that CTA's warps have
all exited.
An MMA's operand tiles and accumulator are read off the layouts it was
derived from (`Core.touch`); every other access is read off the
instruction's registers. `warpc hb ...` prints the happens-before table:
for each buffer, each kind of ordered pair of accesses, and the
synchronisation that first ordered the earlier before the later. The
13 programs `warpc gemm` chooses at the README's 11 shapes, on the two GPUs,
have 87 thousand to 37 million checked operands each, and every one passes:
no operand wrong, no barrier protocol broken, no access that nothing
orders.

`warpc expand` prints a program after expansion, one line an instruction,
registers by name, each derived instruction marked `+` and each construct's
`~`; every unmarked line is one the program wrote. `warpc gemm` compiles
the same list. The launch -- threads, named barriers, shared memory,
barriers, whether the tensor core is used -- is read off the program
(`Mem.finish`), and a kernel is a value: built twice, it compiles to the
same text.

## The layouts the instructions fix

Every derived address, descriptor field and tensor-map box is read off a
layout of `lib/`:

- `backend/atom.ml` holds the layouts the instructions fix, as values of the
  algebra: the K-major SWIZZLE_128B operand a UMMA descriptor describes, the
  SWIZZLE_128B image of a tensor-map box, the tensor-memory accumulator, the
  fragment of `tcgen05.ld.32x32b`. Each comes with a decider that reads the
  instruction's encoding back off a layout and compares the image it
  describes with the layout at every coordinate, so a layout the instruction
  cannot express is refused at compile time.
- The MMA is a value too: `Atom.umma` is CuTe's `SM100_MMA_F16BF16_SS` /
  `_2x1SM_SS` -- operand and accumulator types, M x N over the CTAs it spans,
  its cta_group, the operands' majors and where each is read from -- with the
  shapes CuTe accepts and CuTe's A, B and C thread-value layouts. The rows of
  each operand a CTA stages, the rows of the result it holds, the K step and
  the 32-bit instruction descriptor (field by field, as
  `UMMA::InstrDescriptor`) are read off it.
- The accumulator's way out is a plan (`backend/store.ml`): the load's
  fragment dealt over the accumulator's blocks (`interleave`) composed with
  the accumulator divided into blocks (`divide`), and with the matrix's
  layout or the staging box's. The composites are checked -- injective,
  vectors contiguous and aligned, the load's warp-uniform-address contract
  -- and their strided forms (`Decide`), or their pipelines sliced at the
  compile-time coordinates (`Restricted`), are what `backend/emit.ml` turns
  into instructions. The CTA-to-tile map is a layout too
  (`backend/tiles.ml`).
- The host encodes each tensor map from the `.tmap` lines the kernel emits,
  so it cannot disagree with the addresses the kernel computes.

`backend/test_derive.ml` pins the derivations against what the device and
ptxas use; `backend/test_core.ml` checks that every statement is one
instruction, that colouring keeps live values apart, aligns wide operands
and refuses a program that does not fit, that the derived addresses and
the refusals above hold, that the checker catches a derived address off by
16 bytes and one never computed, that the marks are where they belong, and
that a kernel built twice is the same.

## A program

The tensor-core warp of the 256 x 256 kernel (`backend/gemm_swapped.ml`), in
the configuration `warpc gemm` chooses from 8192 cubed up. The buffers and
the pipe it names are declared once:

    let sa = Mem.ring km ~name:"sa" ~depth ~elem l_sa in       (* A's tiles *)
    let sb = Mem.ring km ~name:"sbt" ~depth ~elem l_sb in      (* B^T's *)
    let stages = Pipe.make km ~name:"stage" ~depth ~fills:1 ~frees:cluster_n in

and, with `inp` the warp's consumer end of the pipe, a stage of the k loop
is

    k_loop (fun kk ->
      Pipe.consume inp (fun () ->
        Prog.emit p (uisetp_ne accum first);
        if ask_ahead then Prog.emit p (bra probe landed);
        Pipe.wait_full inp;
        Prog.label p landed;
        for r = 0 to reps - 1 do                  (* two stacked M = 256 atoms *)
          for j = 0 to (tile_k / mk) - 1 do
            Mem.mma km ~leader regs ~a:(Mem.sub (Pipe.slot_c inp sb) (r * a_rows, j * mk))
              ~b:(Mem.sub (Pipe.slot_c inp sa) (0, j * mk)) ~d:(acc, r * span)
              ~accumulate:(if j = 0 then `If accum else `Always)
          done
        done;
        Prog.emit p (umov first 1);
        Mem.commit2_mc km ~leader (Pipe.empty inp) ~mask:mask_all);
      if ask_ahead then Prog.when_ p more (fun () -> Pipe.probe_full inp ~into:probe))

-- a buffer at the pipe's stage, a sub-tile of it, no address. As `warpc
expand swapped 8192 8192 8192 256 256 4 --clc --ask-ahead --serpentine`
prints it:

    LOOP_3:
    ~ IADD3 k, PT, PT, k, 0x1, RZ
    ~ ISETP.LT.U32.AND more, PT, k, 0x80, PT
    ~ UISETP.NE.U32.AND stage_stays, UPT, stage_stage, 0x3, UPT
    + UIMAD bar@stage, stage_stage, 0x8, smem
    + UIMAD sbt_desc@stage, stage_stage, 0x800, sbt_desc_base
    + UIMAD sa_desc@stage, stage_stage, 0x400, sa_desc_base
    + UIMAD stage_empty@stage, stage_stage, 0x8, stage_empty_base
      UISETP.NE.U32.AND accumulate, UPT, first, URZ, UPT
      @probe BRA READY_1
    WAIT_5:
      SYNCS.PHASECHK.TRANS64.TRYWAIT waited, [bar@stage], stage_parity
    ~ @!waited NANOSLEEP.SYNCS 0xc350
    ~ @!waited BRA WAIT_5
    READY_1:
    + UIADD3 desc_b0.0, UPT, UPT, sa_desc@stage, 0x0, URZ
    + UIADD3 desc_a0.0, UPT, UPT, sbt_desc@stage, 0x0, URZ
      @leader UTCHMMA.2CTA gdesc[desc_a0.0], gdesc[desc_b0.0], tmem[acc], tmem[zero_idesc.0], idesc[zero_idesc.1], accumulate
      ... seven more MMAs, the k steps alternating between two descriptor pairs;
          the second atom's at sbt_desc@stage + 0x400, into acc_col = acc + 0x100
      UMOV first, 0x1
      @leader UTCBAR.2CTA.MULTICAST [stage_empty@stage], URZ, mask
    ~ PLOP3.LUT stage_stays_p, PT, PT, PT, stage_stays, 0x80, 0x8
    ~ @stage_stays UIADD3 stage_stage, UPT, UPT, stage_stage, 0x1, URZ
    ~ @!stage_stays UIADD3 stage_stage, UPT, UPT, stage_stage, -0x3, URZ
    ~ @!stage_stays_p LOP3.LUT stage_parity, stage_parity, 0x80000000, RZ, 0x3c, !PT
    + UIMAD bar@stage, stage_stage, 0x8, smem
      @more SYNCS.PHASECHK.TRANS64.TRYWAIT probe, [bar@stage], stage_parity
    ~ @more BRA LOOP_3

The unmarked lines are the program's: the accumulate flag, the branch past
the wait, the try-wait, the MMAs, the flag, the commit, the probe. The
stage's four addresses are one UIMAD each, where the stage is asked about;
the waits on the full and empty barriers share one (`bar@stage`), each
naming its array by an immediate offset, and the commit that frees the
stage has its own (`stage_empty@stage`). In the SASS warpc writes, the
loop is 45 instructions and the producer's 22.

## The B300

`warpc --sm sm_103a <program>` names the B300 (compute capability 10.3) in
the header, and `warpc lift` makes it the PTX's target.

A kernel launched with the programmatic stream serialization attribute may
begin while the grid before it runs: warpc places PREEXIT at the kernel's
start (or, with `--late-launch`, once the producer's last load is out) and
ACQBULK after the setup -- the barriers initialised, the tensor memory
allocated, the cluster met -- so nothing reads or writes global memory
before the grids ahead have finished. A kernel with a PREEXIT or an ACQBULK
is refused if a load or store of global memory is reached on some path
without an ACQBULK before it.

## Lifting to PTX

`warpc lift k.sass > k.ptx` writes the SASS text as PTX, and ptxas
schedules it, allocates its registers and encodes it: this is how every
kernel here runs. Each instruction becomes the PTX that asks for the same
work, and each value's live range its own register. What differs: the
uniform datapath's single-issue instructions (MMA, commit, copy-engine load
and store) run on the lane `elect.sync` picks at the kernel's entry, and a
uniform guard on one is a branch around it; the allocation construct is one
`tcgen05.alloc`; a try-wait's sleep is the try-wait's suspend-time hint; the
cluster's arrival is `.relaxed`, since the SASS's orders no memory; PREEXIT
and ACQBULK are `griddepcontrol.launch_dependents` and `griddepcontrol.wait`;
the store that adds in the L2 is `cp.reduce.async.bulk.tensor` with `.add`;
the kernel states the CTAs an SM holds by its shared memory
(`.minnctapersm`). Every opcode the 22 programs (the README's 11 shapes,
on the two GPUs) use is one ptxas itself emits.

PTX allows one `.cta_group` in a kernel (PTX ISA 9.0 on): a program that
issues pair MMAs allocates, relinquishes and frees its tensor memory for the
CTA pair (`cta_group::2`), one warp of each CTA taking part, and the lifter
refuses a kernel whose instructions name both groups. ptxas 12.9 and 13.4
build every program.

ptxas 12.8 and 12.9 crash on a `lop3.b32` whose result is tested equal to 0
under a guard; the lifter writes a LUT of two inputs as the `and`, `or`,
`xor` or `not` it is.

## Programs and options

`warpc gemm M N K` compiles the program chosen for the shape. The
programs themselves take their configuration: `warpc one M N K DEPTH` (one
CTA a 128 x 128 tile), `warpc pair M N K DEPTH` (CUTLASS's pair), `warpc
swapped M N K TM TN DEPTH [--clc] [--ask-ahead] [--direct] [--serpentine]
[--cluster-n N] [--bufs B] [--streamk]` (the swapped orientation: tiles by
cluster launch control; asking a stage ahead; each warp storing its registers
straight to C; the tile walk reversing every other band; pairs sharing the
tile of B^T; accumulators; stream-K). `--stamps` and `--stage-stamps` build a
kernel that writes the global timer at each phase boundary, or at each
stage. `warpc expand ...` prints any of them as the program, one line an
instruction, `warpc check ...` checks every derived operand against its
layout, the barriers' protocol and the happens-before order on the host, and
`warpc hb ...` prints the happens-before table. The output is SASS with a
header naming the launch: registers, shared memory, grid, cluster, and a
`.tmap` line per tensor map (`.ptr` for a plain pointer).
`backend/node/run_tma.py k.sass k.cubin` encodes the tensor maps from the
header, launches ptxas's build, checks it against numpy and times it;
`backend/node/ab.py ROUNDS REPEAT a.cubin [b.cubin | cublas] ...` times
several against one another in one process.
