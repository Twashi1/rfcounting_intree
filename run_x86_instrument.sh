#!/usr/bin/env sh

mkdir -p objects
rm -f MBB_stats.csv CritPath.csv reg_stats.csv PathBlocks.csv CFG.csv TopoComp.csv PerBlockAdditional.csv DAG.csv PathCFG.csv

POLY=./PolyBenchC-4.2.1
OBJ=./objects
INCLUDES="-I$POLY/utilities"
FULL_DIRNAME=$(dirname "$1")
# TODO: this line is wrong, but doesn't fail, and changing it messes up paths so 
DIRNAME=${DIRNAME#"$POLY"/}
NAME=$(basename "$1" ".c")
SRC=$1

# 1. Emit IR (O0) per translation unit
./llvm-build/bin/clang -O0 -S -emit-llvm \
  $INCLUDES -I"$POLY/$DIRNAME" \
  "$SRC" \
  -o "$OBJ/$DIRNAME/$NAME.ll"

./llvm-build/bin/clang -O0 -S -emit-llvm \
  $INCLUDES \
  "$POLY/utilities/polybench.c" \
  -o "$OBJ/$DIRNAME/polybench.ll"

# 2. Merge IR for reference
./llvm-build/bin/llvm-link \
  "$OBJ/$DIRNAME/$NAME.ll" \
  "$OBJ/$DIRNAME/polybench.ll" \
  -S -o "$OBJ/$DIRNAME/$NAME.merged.ll"

# 8. Final llc with machine passes
# ./llvm-build/bin/llc \
#   -debug-only=x86-m5-marker\
#   -filetype=obj \
#   "$OBJ/$DIRNAME/$NAME.merged.ll" \
#   -o "$OBJ/$DIRNAME/$NAME.o"

./llvm-build/bin/llc \
  -debug-only=x86-m5-marker\
  -stop-after=x86-m5-marker\
  "$OBJ/$DIRNAME/$NAME.merged.ll" \
  -o - 2> dump.mir

# to get executable: clang -no-pie object file -o output executabel
