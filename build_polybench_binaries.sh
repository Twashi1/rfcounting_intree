#!/usr/bin/env bash
# Commented lines failed to build; missing some dependent math library somewhere
vars=(
"./objects/gemm.o"
"./objects/symm.o"
"./objects/syrk.o"
"./objects/syrk2k.o"
"./objects/gemver.o"
"./objects/2mm.o"
"./objects/3mm.o"
"./objects/atax.o"
#"./PolyBenchC-4.2.1/linear-algebra/solvers/cholesky/cholesky.c"
"./objects/durbin.o"
"./objects/lu.o"
"./objects/ludcmp.o"
"./objects/correlation.o"
"./objects/covariance.o"
#"./PolyBenchC-4.2.1/medley/deriche/deriche.c"
"./objects/floyd-warshall.o"
"./objects/adi.o"
"./objects/jacobi-2d.o"
"./objects/seidel-2d.o"
)

for f in "${vars[@]}"; do
  name="${f%.o}"
  clang -no-pie -lm "$f" -o "$name"
done
