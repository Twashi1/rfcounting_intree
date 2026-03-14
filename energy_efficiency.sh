#!/usr/bin/env bash

program_name=$1

sudo python3 ./scripts/calc_energy_efficiency.py --stats="./stats/${program_name}_mbb_STD.csv" --mcpat_ins="./mcpat_inputs/${program_name}" --mcpat_outs="./mcpat_out/${program_name}" --tei_vf_levels="VoltageFrequency.csv" --file_prefix="${program_name}" --heat_data="./block_heats/${program_name}_ProgramHeat.csv" --baseline_heat="./block_heats/${program_name}_Baseline_ProgramHeat.csv"
