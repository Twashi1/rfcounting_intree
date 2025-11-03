#!/bin/bash

mkdir -p objects

./llvm-build/bin/clang -O2 -S -emit-llvm \
  -I./PolyBenchC-4.2.1/utilities/ \
  -I./PolyBenchC-4.2.1/linear-algebra/blas/gemm \
  -c ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c \
  -o ./objects/gemm.ll
# debug only to print out any dbgs() << 
./llvm-build/bin/llc -debug-only=reg-access-postra,reg-access-prera objects/gemm.ll -o objects/gemm.s
