#!/usr/bin/env sh

NAME=$(basename "$1" ".c")
SRC=$1

mkdir -p ./testcode/objects/

# 1. Emit IR (O0) per translation unit
./llvm-build/bin/clang -O3 -S -emit-llvm \
  "$SRC" \
  -o "./testcode/objects/$NAME.ll"

./llvm-build/bin/llc -debug-only=thermal -misched=thermal "./testcode/objects/$NAME.ll"
