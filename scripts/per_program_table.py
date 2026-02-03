import argparse
import utils

# NOTE: we removed L1 caches, L2 cache, Translation Lookaside Buffers
# TODO: maybe some more of these should be excluded? like the queues IntQ, FPQ, LdStQ?
HOTSPOT_CORE_FLOORPLAN = {
    "Bpred_0",
    "Bpred_1",
    "Bpred_2",
    "FPAdd_0",
    "FPAdd_1",
    "FPReg_0",
    "FPReg_1",
    "FPReg_2",
    "FPReg_3",
    "FPMul_0",
    "FPMul_1",
    "FPMap_0",
    "FPMap_1",
    "IntMap",
    "IntQ",
    "IntReg_0",
    "IntReg_1",
    "IntExec",
    "FPQ",
    "LdStQ",
}


def main():
    # TODO: load HeatData.csv
    # TODO: load output if exists
    # TODO: get max of functional units, get mean of functional units, get weighted on width/height
    # TODO:
    parser = argparse.ArgumentParser(
        description="Get average, and max heat data per functional unit"
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()
