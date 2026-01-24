#!/bin/sh

# Generate mcpat files per dvs calling point/block
# Just keeping this command for safekeeping at the last, can generalise later
python3 ./scripts/create_mcpat_per_dvs.py --stats=./stats/gemm_path_STD.csv --output_xml=gemm --output_dir=./mcpat_inputs/gemm
