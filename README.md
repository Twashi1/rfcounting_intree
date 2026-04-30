# Compile-time analysis for DVS scaling

Repository should be cloned recursively, however note that gem5 validation and HotSpot 6.0 have not been packaged with this repository.

The project does expect HotSpot 6.0 at the path `./../HotSpot/<binary>`

Required python packages
- matplotlib
- numpy
- pandas
- seaborn

A full run should be possible under the following steps
1. `./buildscript.sh` (will require many libraries for building LLVM from scratch, cannot list them all here)
2. `./run_workflow_all_polybench_path.sh` (may take very long on slower systems)

If looking at the code is all that is required, the following directories contains the bulk of the code
- `./scripts/`; majority of Python scripts that link the various tools, and perform the optimisation loop.
- `./llvm-project/llvm/lib/CodeGen/RegisterAccessPreRAPass/RegisterAccessPreRAPass.c`; The LLVM analysis pass including the statistics collection and path-clubbing algorithm
- `./llvm-project/llvm/include/llvm/CodeGen/RegisterAccessPreRAPass.h`; Corresponding header for above
