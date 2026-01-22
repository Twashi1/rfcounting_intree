#!/bin/sh

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$ROOT_DIR/build"
LLVM_BUILD="$ROOT_DIR/../llvm-build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DLLVM_DIR=/usr/lib/llvm-21 \
  -DBUILD_SHARED_LIBS=OFF \
  ..

cmake --build .
