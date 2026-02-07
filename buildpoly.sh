#!/usr/bin/env sh

mkdir -p objects
rm -f CritPath.csv
rm -f MBB_stats.csv
rm -f reg_stats.csv

#./llvm-build/bin/clang -O2 -S -emit-llvm \
# -I./PolyBenchC-4.2.1/utilities/ \
# -I./PolyBenchC-4.2.1/linear-algebra/blas/gemm \
# -c ./PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c \
# -o ./objects/gemm.ll

POLY=./PolyBenchC-4.2.1
OBJ=./objects
INCLUDES="-I$POLY/utilities"
LOGFILE="./CompiledFiles.txt"
DEFAULT_K=1000
# number of files
K=${1:-$DEFAULT_K}
# clear file
: > "$LOGFILE"

BENCHMARKS=$(find "$POLY" -type f -name "*.c" -not -path "*/utilities/*" | head -n "$K")

for SRC in $BENCHMARKS; do
  NAME=$(basename "$SRC" .c)
  RELPATH=${SRC#$POLY/}
  DIRNAME=$(dirname "$RELPATH")

  # skip non-benchmark files
  if [ "$NAME" = "polybench" ]; then
    continue
  fi

  echo "Building $NAME"

  mkdir -p "$OBJ/$DIRNAME"

  ./llvm-build/bin/clang -O2 -S -emit-llvm -I"$POLY/utilities" -I"$POLY/$DIRNAME" \
    "$SRC" \
    -o "$OBJ/$DIRNAME/$NAME.ll"

  echo "Running LLC on $NAME"

  ./llvm-build/bin/llc -debug-only=reg-access-prera,reg-access-postra "$OBJ/$DIRNAME/$NAME.ll" \
    -o "$OBJ/$DIRNAME/$NAME.s"

  echo "$SRC" >> "$LOGFILE"
done

# debug only to print out any dbgs() << 
# ./llvm-build/bin/llc -debug-only=reg-access-postra,reg-access-prera objects/gemm.ll -o objects/gemm.s
