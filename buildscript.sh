#!/usr/bin/env bash

cmake -G Ninja \
  -S llvm-project/llvm \
  -B llvm-build \
  -DLLVM_ENABLE_PROJECTS="llvm;clang" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCOMPILER_RT_BUILD_XRAY=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_USE_RELATIVE_PATHS=OFF \
  -DCMAKE_BUILD_TYPE=Release

ninja -C llvm-build

# we don't use compiler-rt but ill keep it here
