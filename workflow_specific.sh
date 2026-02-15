#!/usr/bin/env bash
polybench_path=$1
test_name=$(basename "$1" .c)

# NOTE: excluding buildscript, assume already built
# sudo sh ./buildscript.sh
sudo bash ./buildpoly_specific.sh "${polybench_path}"

# Create standard stat file
rm -f ./stats/${test_name}_mbb_STD.csv ./stats/${test_name}_path_STD.csv

python3 ./scripts/create_stats.py --input_file=MBB_stats.csv --output=./stats/${test_name}_mbb --module_index=2 --take_sum=0
#python3 ./scripts/create_stats.py --input_file=PathBlocks.csv --output=./stats/${test_name}_path --module_index=2 --path_index=-1

# Set initial voltages to v4 (rough middle)
sudo python3 ./scripts/initial_voltages.py --voltage_level=4 --module_index=2
# Delete old McPAT inputs
sudo rm -rf "./mcpat_inputs/${test_name}"
sudo python3 ./scripts/create_mcpat_per_block.py --stats="./stats/${test_name}_mbb_STD.csv" --output_xml="${test_name}" --output_dir="./mcpat_inputs/${test_name}"

# Delete old McPAT outputs
sudo rm -rf "./mcpat_out/${test_name}"
# Feed all mcpat inputs
sudo sh ./run_mcpat_all.sh "./mcpat_inputs/${test_name}"

# Feed outputs into ptrace to generate final heats
sudo python3 ./scripts/mcpat_to_ptrace.py --mcpat_outs="./mcpat_out/${test_name}" --module_index=2 --aggregate=likely

# Generate heat data table
sudo python3 ./scripts/per_program_table.py --name="${test_name}"

# TODO: inserting DVS calls (gem5 now, not sniper)
# TODO: lower IR to machine code
# TODO: test in gem5
