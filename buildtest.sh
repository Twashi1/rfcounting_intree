#!/bin/bash

filename=$(basename "$1")

mkdir -p objects

./llvm-build/bin/clang -O2 -S -emit-llvm $1 -o objects/$filename.ll
# debug only to print out any dbgs() << 
./llvm-build/bin/llc -debug-only=reg-access-postra,reg-access-prera objects/$filename.ll -o objects/$filename.s


