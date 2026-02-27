#!/usr/bin/env bash

test_name=$(basename "$1" .c)

# Calculate EDP
sudo python3 ./scripts/initial_voltages.py --n=100 --voltage_level=4
echo "Test name: ${test_name}" >> "efficiencyStats.txt"
sudo python3 ./scripts/calc_energy_efficiency.py --stats="./stats/${test_name}_mbb_STD.csv" --mcpat_outs="./mcpat_out/${test_name}" --old_voltage_levels="VoltageLevels.csv" --new_voltage_levels="./block_heats/${test_name}_OutVoltages.csv" >> "efficiencyStats.txt"
