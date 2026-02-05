#!/usr/bin/env bash

x="$1"
# Kelvin conversion
k=$(awk -v x="$x" 'BEGIN { printf "%.2f", x + 273.15 }')

# Units to give new values to
units=(
L2_left L2 L2_right Icache Dcache
Bpred_0 Bpred_1 Bpred_2
DTB_0 DTB_1 DTB_2
FPAdd_0 FPAdd_1
FPReg_0 FPReg_1 FPReg_2 FPReg_3
FPMul_0 FPMul_1
FPMap_0 FPMap_1
IntMap IntQ IntReg_0 IntReg_1 IntExec
FPQ LdStQ ITB_0 ITB_1
)

prefixes=("" "iface_" "hsp_ "hsink_")

for p in "${prefixes[@]}"; do
  for u in "${units[@]}"; do
    printf '%s%s\t%.2f\n' "$p" "$u" "$k"
  done
done

for i in {0..11}; do
  printf 'inode_%d\t%.2f\n' "$i" "$k"
done
