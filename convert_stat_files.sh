#!/bin/sh

python3 ./scripts/create_stats.py --input_file=MBB_stats.csv --output=./stats/$1_mbb --module_index=2
python3 ./scripts/create_stats.py --input_file=PathBlocks.csv --output=./stats/$1_path0 --module_index=2 --path_index=0

python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_low_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=low
python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_med_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=med
python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_high_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=high

python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_low_path0.xml --stats=./stats/$1_path0_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=low
python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_med_path0.xml --stats=./stats/$1_path0_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=med
python3 ./scripts/create_mcpat.py --input_xml=./mcpat_inputs/Alpha21364.xml --output_xml=./mcpat_inputs/$1_high_path0.xml --stats=./stats/$1_path0_STD.csv --input_cfg=./mcpat_inputs/mcpat.cfg --voltage_level=high
