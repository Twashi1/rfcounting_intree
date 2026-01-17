mkdir -p objects
rm -f MBB_stats.csv CritPath.csv reg_stats.csv PathBlocks.csv

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

# 3. Build instrumented executable
./llvm-build/bin/clang -O0 -fprofile-instr-generate \
  $INCLUDES -I"$POLY/$DIRNAME" \
  "$SRC" "$POLY/utilities/polybench.c" \
  -no-pie \
  -o "$OBJ/$DIRNAME/$NAME.instr"

# 4. Run to collect profile
LLVM_PROFILE_FILE="$OBJ/$DIRNAME/$NAME.profraw" \
  "$OBJ/$DIRNAME/$NAME.instr"

# 5. Merge profile
./llvm-build/bin/llvm-profdata merge \
  "$OBJ/$DIRNAME/$NAME.profraw" \
  -o "$OBJ/$DIRNAME/$NAME.profdata"

# 6. Apply profile per translation unit
./llvm-build/bin/clang -O2 -fprofile-use="$OBJ/$DIRNAME/$NAME.profdata" \
  -S -emit-llvm \
  $INCLUDES -I"$POLY/$DIRNAME" \
  "$SRC" \
  -o "$OBJ/$DIRNAME/$NAME.pgo.ll"

./llvm-build/bin/clang -O2 -fprofile-use="$OBJ/$DIRNAME/$NAME.profdata" \
  -S -emit-llvm \
  $INCLUDES \
  "$POLY/utilities/polybench.c" \
  -o "$OBJ/$DIRNAME/polybench.pgo.ll"

# 7. Merge PGO IR
./llvm-build/bin/llvm-link \
  "$OBJ/$DIRNAME/$NAME.pgo.ll" \
  "$OBJ/$DIRNAME/polybench.pgo.ll" \
  -S -o "$OBJ/$DIRNAME/$NAME.pgo.merged.ll"

# 8. Final llc with machine passes
./llvm-build/bin/llc \
  -debug-only=reg-access-prera,reg-access-postra \
  -filetype=obj \
  "$OBJ/$DIRNAME/$NAME.pgo.merged.ll" \
  -o "$OBJ/$DIRNAME/$NAME.o"

