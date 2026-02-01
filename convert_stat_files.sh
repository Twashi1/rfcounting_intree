#!/bin/sh

CONFIG_FILE=./scripts/configs.cfg
INPUT_XML=./mcpat_inputs/Alpha21364.xml

rm -f ./stats/$1_mbb_STD.csv ./stats/$1_path_STD.csv

python3 ./scripts/create_stats.py --input_file=MBB_stats.csv --output=./stats/$1_mbb --module_index=2 --take_sum=0
python3 ./scripts/create_stats.py --input_file=PathBlocks.csv --output=./stats/$1_path --module_index=2 --path_index=-1

python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_low_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=low
python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_med_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=med
python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_high_mbb.xml --stats=./stats/$1_mbb_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=high

python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_low_path.xml --stats=./stats/$1_path_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=low
python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_med_path.xml --stats=./stats/$1_path_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=med
python3 ./scripts/create_mcpat.py --input_xml="$INPUT_XML" --output_xml=./mcpat_inputs/$1_high_path.xml --stats=./stats/$1_path_STD.csv --input_cfg="$CONFIG_FILE" --voltage_level=high
