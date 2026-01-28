#!/bin/sh

indir="$1"
last_dir=$(basename "${indir%/}")

mkdir -p ./mcpat_out
mkdir -p "./mcpat_out/$last_dir"

echo "Outputting to ./mcpat_out/$last_dir"

for file in "$indir"/*; do
  [ -f "$file" ] || continue
  base=$(basename "$file")
  name="${base%.*}"
  ./McPAT_monolithic/mcpat -infile "$file" -print_level 5 > "./mcpat_out/$last_dir/$name.txt"
  echo "Created ./mcpat_out/$last_dir/$name.txt"
done
