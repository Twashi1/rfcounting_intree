#!/usr/bin/env sh

sudo python3 ./scripts/gem5_analysis.py --input_file="../../Documents/gem5/m5out/stats_run0.txt" --output_folder="./mcpat_out/gem5_gemm" --program_name="gem5_gemm"
