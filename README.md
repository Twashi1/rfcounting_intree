# Compile-time analysis for DVS scaling

## Shell scripts

- `sh ./buildscript.sh` - Build LLVM-project
- `bash ./buildpoly_specific.sh <polybench source>` - Build and run a specific PolyBench test.
    - Example input: `./buildpoly_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c`
- `sh ./convert_stat_files.sh <output_filename>` - Convert output filenames to standard stats with the given output name

## Statistics

Breakdown of the column names of various statistics the program collects.

### Getting statistics

Running `buildpoly_specific.sh <source file>` to build a Polybench test. E.g. `./buildpoly_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c`. Will output the following files
- `PathBlocks.csv`, a breakdown of the stats per basic block of a given module and path
- `MBB_stats.csv`, stats per machine-basic block

Use `scripts/utils.py` for loading files, either with `load_arbitrary_stat_file`.
- `path` the path to the stat file to process.
- `module_index` refers to the module/file you want the statistics for.
- `path_index` refers to the path or basic block to collect stats for. Keep as default or `-1` to sum stats over all basic blocks. 
Possible inputs are:
- `PathBlocks.csv` (specify `path_index`)
- `MBB_stats.csv`
- `gem5_outs/outdir_xxx/stats.txt` (any gem5 output)

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

### Simulator command insertion input `SIM_COMMAND_INPUT.csv`

`module_name,function_name,block_name,is_entry,is_exit,frequency`


