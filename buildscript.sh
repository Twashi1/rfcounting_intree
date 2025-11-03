#!/bin/bash

cmake -G Ninja \
  -S llvm-project/llvm \
  -B llvm-build \
  -DLLVM_ENABLE_PROJECTS="llvm;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release

ninja -C llvm-build
