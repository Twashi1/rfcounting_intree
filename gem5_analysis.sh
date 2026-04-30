#!/usr/bin/env sh

# NOTE: this script was intended to evaluate the performance of an implementation of the state-of-the-art approach, however the implementation yielded worse results than baseline, and thus I didn't think it would be right to include it as a point of comparison; doing nothing would've "beaten" state-of-the-art.

sudo python3 ./scripts/gem5_analysis.py --input_file_dvfs="../../Documents/gem5/m5out/stats_covariance_dvfs0.txt" --input_file_base="../../Documents/gem5/m5out/stats_covariance_base0.txt" --output_folder="./mcpat_out/gem5_covariance" --program_name="gem5_covariance"
