#!/usr/bin/env bash
set -euo pipefail

# Path to your pass plugin
PASS_PLUGIN="./build/AddSimCommandsPass.so"

# Output directories
OBJ_DIR="./objects"

mkdir -p "$OBJ_DIR"

# TODO: note we already have these .ll files, gemm.pgo.merged.ll or whatever, so we'll use that in future
# 1. Compile each C file to LLVM IR (.ll)
# TODO: regardless of above point, shouldn't this be compiling all srcs together
#   instead of each separately?
IR_FILES=()
for src in "$@"; do
    base=$(basename "$src" .c)
    ll="$OBJ_DIR/${base}.ll"
    clang -O2 -S -emit-llvm "$src" -include "test/sim_api.h" -o "$ll"
    IR_FILES+=("$ll")
done

# IR_FILES+=("$OBJ_DIR/sim_cmd.ll")

# 2. Merge LLVM IR into a single file if needed
MERGED_LL="$OBJ_DIR/merged.ll"
if [ "${#IR_FILES[@]}" -gt 1 ]; then
    llvm-link-21 "${IR_FILES[@]}" -o "$MERGED_LL"
else
    cp "${IR_FILES[0]}" "$MERGED_LL"
fi


FILENAME=merged

opt-21 -load-pass-plugin "$PASS_PLUGIN" -passes="function(sim-commands-pass)" $MERGED_LL -S -o "$OBJ_DIR/$FILENAME.ll"

# 4. Optionally generate assembly
llc-21 "$OBJ_DIR/$FILENAME.ll" -o "$OBJ_DIR/$FILENAME.s"

# 5. Optionally generate object file
llc-21 "$OBJ_DIR/$FILENAME.ll" -filetype=obj -o "$OBJ_DIR/$FILENAME.o"
