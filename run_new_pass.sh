#!/usr/bin/env bash

# Commented lines failed to build; missing some dependent math library somewhere
vars=(
"./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c"
"./PolyBenchC-4.2.1/linear-algebra/blas/symm/symm.c"
"./PolyBenchC-4.2.1/linear-algebra/blas/syrk/syrk.c"
"./PolyBenchC-4.2.1/linear-algebra/blas/syrk2k/syrk2k.c"
"./PolyBenchC-4.2.1/linear-algebra/blas/gemver/gemver.c"
"./PolyBenchC-4.2.1/linear-algebra/kernels/2mm/2mm.c"
"./PolyBenchC-4.2.1/linear-algebra/kernels/3mm/3mm.c"
"./PolyBenchC-4.2.1/linear-algebra/kernels/atax/atax.c"
#"./PolyBenchC-4.2.1/linear-algebra/solvers/cholesky/cholesky.c"
"./PolyBenchC-4.2.1/linear-algebra/solvers/durbin/durbin.c"
"./PolyBenchC-4.2.1/linear-algebra/solvers/lu/lu.c"
"./PolyBenchC-4.2.1/linear-algebra/solvers/ludcmp/ludcmp.c" # TODO: seems like we get an error attempting to run this binary?
"./PolyBenchC-4.2.1/datamining/correlation/correlation.c"
"./PolyBenchC-4.2.1/datamining/covariance/covariance.c"
#"./PolyBenchC-4.2.1/medley/deriche/deriche.c"
"./PolyBenchC-4.2.1/medley/floyd-warshall/floyd-warshall.c"
"./PolyBenchC-4.2.1/stencils/adi/adi.c"
"./PolyBenchC-4.2.1/stencils/jacobi-2d/jacobi-2d.c"
"./PolyBenchC-4.2.1/stencils/seidel-2d/seidel-2d.c"
)


for f in "${vars[@]}"; do
  sudo sh ./buildpoly_specific.sh $f
  # Get the filename from the path, then remove the .c extension.
  # For example:
  # ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c
  # becomes:
  # gemm
  program_name=$(basename "$f" .c)

  # Create output directory.
  output_dir="output_stats/$program_name"
  sudo mkdir -p "$output_dir"

  # Move generated statistics into the program-specific directory.
  sudo mv EfficiencyStatsNew.txt "$output_dir/"
  sudo mv DVSInsertionData.csv "$output_dir/"
  sudo mv PerSubgraphStats.csv "$output_dir/"
done
