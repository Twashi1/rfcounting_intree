#!/bin/bash
sudo sh ./buildscript.sh
sudo bash ./buildpoly_specific.sh ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c

# Create standard stat file
sudo sh ./convert_stat_files.sh gemm

# Delete output dir for this
sudo rm -rf ./mcpat_inputs/gemm
sudo python3 ./scripts/create_mcpat_per_block.py --stats=./stats/gemm_mbb_STD.csv --output_xml=gemm --output_dir=./mcpat_inputs/gemm

# Feed all mcpat inputs
sudo sh ./run_mcpat_all.sh ./mcpat_inputs/gemm

# Feed outputs into ptrace to generate final heats
sudo python3 ./scripts/mcpat_to_ptrace.py --mcpat_outs=./mcpat_out/gemm --module_index=2 --aggregate=likely

# TODO: inserting DVS calls (gem5 now, not sniper)
# TODO: lower IR to machine code
# TODO: test in gem5
