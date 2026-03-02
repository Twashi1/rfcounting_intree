#!/usr/bin/env sh

# NOTE: assuming input file without .xml, we add that
inputfile="$1"
inprogram="$2"

mkdir -p ./mcpat_out
mkdir -p "./mcpat_out/${inprogram}"

echo "Outputting to ./mcpat_out/${inprogram}"

./McPAT_monolithic/mcpat -infile "./mcpat_inputs/${inprogram}/${inputfile}.xml" -print_level 5 > "./mcpat_out/${inprogram}/${inputfile}.txt"
echo "Created ./mcpat_out/${inprogram}/${inputfile}.txt"
