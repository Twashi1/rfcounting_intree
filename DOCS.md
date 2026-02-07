# Program/workflow documentation

## Shell scripts

- `buildscript.sh` - Build LLVM-project
- `buildpoly_specific.sh <polybench source>` - Build and run a specific PolyBench test.
    - Example input: `./buildpoly_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c`
- `convert_stat_files.sh <output_filename>` - Convert output filenames to standard stats with the given output name
    - Example input: `./convert_stat_files.sh gemm`
    - Note this will take an average over basic blocks to compute per-basic-block stats
    - And it will take each program path (assumed DVS calling point), and compute stats per path
    - If you want McPAT inputs per basic-block, follow procedure in `workflow_specific.sh`
- `run_mcpat_all.sh <input_folder>` - Runs a folder of McPAT XML inputs through McPAT to get resulting power
    - Example input: `./run_mcpat_all.sh ./mcpat_inputs/gemm`
    - Generates the corresponding output folder at `./mcpat_out/gemm`
- `workflow_specific.sh <polybench source>` - Performs (almost) the full workflow, from a PolyBench source, to analysis, stat estimation, McPAT power trace generation, and thermal trace estimation
    - Example input: `./workflow_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c`
    - Reads from `VoltageLevels.csv` to determine the voltage level to assume for each basic block (expected either low, med, or high)


## General workflow

Generally, build the LLVM analysis pass with `buildscript.sh`. From there, simply use `workflow_specific.sh` to run everything from start to finish for some particular PolyBench source.

0. Build the LLVM analysis pass with `buildscript.sh`
1. Statistics are generated using `buildpoly_specific.sh <polybench source>`
2. Missing statistics are estimated using `scripts/create_stats.py` (no good automated workflow yet)
3. Filled in statistics (notated by having the appended `_STD`, usually present in `stats`) are then converted to PolyBench XML input by `scripts/create_mcpat_xxx.py`. This will depend on converting a single stat instance, or stats per basic block/DVS calling point
4. Feed the crafted McPAT input stats into McPAT to get power traces, using `run_mcpat_all.sh`
5. Generate temperature traces by using CFG information, power traces, execution time, using `scripts/mcpat_to_ptrace.py`. This generated `HeatData.csv`
6. Final heat data per block is transformed into max/mean temperatures across core units by `scripts/per_program_table.py`. Finally generates `ProgramHeat.csv`, prefixed by program name

Note this is not the extent of the planned workflow, as some pre-processing and post-processing stages are still planned.
- Detecting and breaking up large DVS loop
- Insertion of DVS instructions based on heat, aiming to maximise TEI and minimise SHEs
- Automated testing using Gem5 to validate accuracy of temperature estimates, power efficiency benefits, etc.

## Statistics

Breakdown of the column names of various statistics the program collects.

### Getting statistics

Running `buildpoly_specific.sh <source file>` to build a Polybench test. E.g. `./buildpoly_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c`. Will output the following files
- `PathBlocks.csv`, a breakdown of the stats per basic block of a given module and path
- `MBB_stats.csv`, stats per machine-basic block
- `CFG.csv`, the control-flow graph of basic blocks, including connections between functions
- `DAG.csv`, the directed acylic graph representing strongly connected components of the CFG. Used to generate approximately topologically sorted order for basic blocks
- `PerBlockAdditional.csv`, additional per block data, such as execution time, the component in the DAG
- `TopoComp.csv`, topologically sorted components of the DAG
- `HeatData.csv`, per-unit heat in kelvin, per basic block
- `VoltageLevels.csv`, the voltage to assign to each basic block in runs of estimating thermal traces

### Locators and identifiers

- `module_name`: Name of the module being analysed (usually the file)
- `function_name`: The function being analysed
- `block_name`: The specific basic block being analysed

#### Specific to subgraph/path analysis

- `path_index`: The index of the subgraph/path being analysed
- `is_entry`: Whether this block is an entry block for the given `path_index`
- `is_exit`: Whether this block is an exit block for the given `path_index`

### Required stats

- `cycle_count`: Total number of clock cycles executed
- `instr_count`: Total number of instructions executed
- `int_instr_count`: Total number of integer instructions executed
- `float_instr_count`: Total number of floating-point instructions executed
- `branch_instr_count`: Total number of branch instructions executed
- `loads`: Total number of loads executed
- `stores`: Total number of stores executed
- `freq`: Expected number of times block will run relative to function entrypoint
- `int_regfile_reads`: Number of integer register reads
- `int_regfile_writes`: Number of integer register writes
- `float_regfile_reads`: Number of float regfile reads
- `float_regfile_writes`: Number of float regfile writes
- `function_calls`: Function calls
- `context_switches`: Context switches, returns and jumps
- `mul_access`: Multiplier unit accesses
- `fp_access`: Floating point unit accesses
- `ialu_access`: Integer ALU accesses

### Estimated stats

These stats can be estimated from others (although likely poorly).

- `idle_cycles`: Cycles clock was idle for (assumed 0)
- `busy_cycles`: Cycles clock was running for (assumed `cycle_count`)
- `committed_instr`: Instructions commited to CPU? (assumed `instr_count`)
- `committed_int_instr`: Int instructions commited to CPU? (assumed `int_instr_count`)
- `committed_float_instr`: Float instructions commited to CPU? (assumed `float_instr_count`)
- `branch_mispredictions`: Number of branch mispredicts (assumed 0)
- `rob_reads`: Reorder buffer reads (assumed `instr_count`)
- `rob_writes`: Reorder buffer writes (assumed `instr_count`)
- `rename_reads`: Rename reads (assumed `instr_count * 2`)
- `rename_writes`: Rename writes (assumed `instr_count`)
- `fp_rename_reads`: Floating-point rename reads (assumed `float_instr_count * 2`)
- `fp_rename_writes`: Floating-point rename writes (assumed `float_instr_count`)
- `inst_window_writes`: ? (assumed `instr_count`)
- `inst_window_reads`: ? (assumed `instr_count`)
- `inst_window_wakeup_acccesses`: ? (assumed `inst_window_writes + instr_window_reads`)
- `fp_inst_window_reads`: ? (assumed `float_instr_count`)
- `fp_inst_window_writes`: ? (assumed `float_instr_count`)
- `fp_inst_window_wakeup_accesses`: ? (assumed `fp_inst_window_reads + fp_inst_window_writes`)
- `cdb_mul_accesses`: ? (assumed `mul_access`)
- `cdb_alu_accesses`: ? (assumed `ialu_access`)
- `cdb_fp_accesses`: ? (assumed `fp_access`)
- `btb_reads`: ? (assumed `instr_count`)
- `btb_writes`: ? (assumed 0)

## CSV Files



