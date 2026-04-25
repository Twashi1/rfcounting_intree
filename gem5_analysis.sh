#!/usr/bin/env sh

sudo python3 ./scripts/gem5_analysis.py --input_file_dvfs="../../Documents/gem5/m5out/stats_covariance_dvfs0.txt" --input_file_base="../../Documents/gem5/m5out/stats_covariance_base0.txt" --output_folder="./mcpat_out/gem5_covariance" --program_name="gem5_covariance"
