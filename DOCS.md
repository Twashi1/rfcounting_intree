# Program/workflow documentation

## Important files 

- `buildscript.sh` - Build LLVM-project
- `run_workflow_all_polybench_path.sh` - Runs a list of PolyBench benchmarks, selecting optimal voltage-frequency configurations and calculating the improved efficiency of these configurations. Primary output is `efficiencyStats.txt`, however additional temperature and DVFS calling counts can be found in the `./block_heats` directory
    - Within this file are the variables `VAR_FREQUENCY` to switch between targetting stable-frequency operation or variable-frequency operation
    - `NO_DELETE` prevents deletion of the McPAT power traces, which consumes the majority of processing time
- `./scripts/configs.cfg` - Main configuration options; voltage levels, frequency range, assumed temperatures, tech node size

## Required stats

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

## Estimated stats

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



