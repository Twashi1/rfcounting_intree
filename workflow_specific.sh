#!/usr/bin/env bash
polybench_path=$1
test_name=$(basename "$1" .c)

# NOTE: excluding buildscript, assume already built
# sudo sh ./buildscript.sh
sudo bash ./buildpoly_specific.sh "${polybench_path}"

# Create standard stat file
rm -f ./stats/${test_name}_mbb_STD.csv ./stats/${test_name}_path_STD.csv

python3 ./scripts/create_stats.py --input_file=MBB_stats.csv --output=./stats/${test_name}_mbb --module_index=2 --take_sum=0

# TODO: note this shouldn't be used in any meaningful way, should remove soon
sudo python3 ./scripts/initial_voltages.py --voltage_level=5 --module_index=2
# Delete old McPAT inputs
sudo rm -rf "./mcpat_inputs/${test_name}"
sudo python3 ./scripts/create_mcpat_per_block.py --stats="./stats/${test_name}_mbb_STD.csv" --output_xml="${test_name}" --output_dir="./mcpat_inputs/${test_name}"

# Delete old McPAT outputs
sudo rm -rf "./mcpat_out/${test_name}"
# Feed all mcpat inputs
# sudo sh ./run_mcpat_all.sh "./mcpat_inputs/${test_name}"

# Feed outputs into ptrace to generate final heats
sudo python3 ./scripts/mcpat_to_ptrace.py --mcpat_outs="./mcpat_out/${test_name}" --module_index=2 --aggregate=likely --mcpat_ins="./mcpat_inputs/${test_name}" --stats="./stats/${test_name}_mbb_STD.csv"

# Generate heat data table
sudo python3 ./scripts/per_program_table.py --name="./block_heats/${test_name}" --stats="./stats/${test_name}_mbb_STD.csv"
sudo python3 ./scripts/per_program_table.py --heatdata="HeatDataBaseline.csv" --name="./block_heats/${test_name}_Baseline" --stats="./stats/${test_name}_mbb_STD.csv"

# Add required voltages
sudo python3 ./scripts/tei_effects.py --program_heat="./block_heats/${test_name}_ProgramHeat.csv" --out_prefix="./block_heats/${test_name}"

# Write voltages to VoltageLevels.csv
sudo python3 ./scripts/read_voltages.py --tei_voltages="./block_heats/${test_name}_ProgramHeatVoltages.csv" --out_voltages="./block_heats/${test_name}_OutVoltages.csv"

# Calculate EDP
sudo python3 ./scripts/calc_energy_efficiency.py --stats="./stats/${test_name}_mbb_STD.csv" --mcpat_ins="./mcpat_inputs/${test_name}" --mcpat_outs="./mcpat_out/${test_name}" --new_voltage_levels="./block_heats/${test_name}_OutVoltages.csv" --file_prefix="${test_name}"

# Calculate temperature difference
sudo python3 ./scripts/temp_difference.py --etc_heat="./block_heats/${test_name}_ProgramHeat.csv" --baseline_heat="./block_heats/${test_name}_Baseline_ProgramHeat.csv" --out_prefix="./block_heats/${test_name}"
