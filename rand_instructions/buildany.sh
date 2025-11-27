#!/bin/bash

mkdir -p objects
name=$(basename "$1" .cpp)
g++ -static -O2 $1 -o ./objects/$name-x86
