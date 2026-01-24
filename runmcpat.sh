#!/bin/sh

base=$(basename "$1")
name="${base%.*}"

./McPAT_monolithic/mcpat -infile "$1" -print_level 5 > "./mcpat_out/$name.txt"

echo "Created ./mcpat_out/$name.txt"
