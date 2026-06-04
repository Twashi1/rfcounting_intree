#!/usr/bin/env sh

POLY=./PolyBenchC-4.2.1
OBJ=./objects
INCLUDES="-I$POLY/utilities"
FULL_DIRNAME=$(dirname "$1")
# TODO: this line is wrong, but doesn't fail, and changing it messes up paths so 
DIRNAME=${DIRNAME#"$POLY"/}
NAME=$(basename "$1" ".c")
SRC=$1

# 1. Emit IR (O0) per translation unit
./llvm-build/bin/clang -O3 -S -emit-llvm \
  $INCLUDES -I"$POLY/$DIRNAME" \
  "$SRC" \
  -o "$OBJ/$DIRNAME/$NAME.ll"

./llvm-build/bin/clang -O3 -S -emit-llvm \
  $INCLUDES \
  "$POLY/utilities/polybench.c" \
  -o "$OBJ/$DIRNAME/polybench.ll"

# 2. Merge IR for reference
./llvm-build/bin/llvm-link \
  "$OBJ/$DIRNAME/$NAME.ll" \
  "$OBJ/$DIRNAME/polybench.ll" \
  -S -o "$OBJ/$DIRNAME/$NAME.merged.ll"

./llvm-build/bin/llc -debug-only=thermal -misched=thermal "$OBJ/$DIRNAME/$NAME.merged.ll"
