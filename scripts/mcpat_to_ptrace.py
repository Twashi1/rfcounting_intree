# TODO: read in some folder of mcpat .txt inputs, and convert to ptrace for HotSpot
# TODO: also requires the HotSpot floorplan
import utils
import argparse
import re
import pandas as pd

# TODO: we need CFG data
#   given some single mcpat file, we generate a single row of ptrace (a single sample)
#   our sampling window is not-constant
#   so instead, we take the output temps, and feed them back in as the initial temps?


# TODO: "name" is actually name prefix
def mcpat_get_unit_stats(name: str, text: str) -> dict:
    # Note sometimes a colon, sometimes not
    pattern = rf"\s*{re.escape(name)}.*\n(\s+[\s\w=]+\s*([\d\.])+.*\n)+"

    m = re.search(pattern, text, re.MULTILINE)

    if not m:
        return None

    block = m.group(1)
    fields = {}

    for key in [
        "Area",
        "Peak Dynamic",
        "Subthreshold Leakage",
        "Gate Leakage",
        "Runtime Dynamic",
    ]:
        fm = re.search(rf"{key}\s*=\s*([0-9.eE+-]+)", block)
        if fm:
            fields[key] = float(fm.group(1))

    print(fields)

    return fields


def mcpat_to_dict(mcpat_out_file: str) -> dict:
    text = ""

    with open(mcpat_out_file, "r") as f:
        text = f.read()

    # TODO: note this is just a subset of the actual things we need for the floormap, and we probably need some renames
    unit_stats = {}

    for unit in [
        "L2",
        "Instruction Cache",
        "Data Cache",
        "Branch Predictor",
        "Dtlb",
        "Floating Point Unit",
        "Floating Point RF",
        "Renaming Unit",
        "Int Front End RAT",
        "FP Front End RAT",
        "Int Retire RAT",
        "FP Retire RAT",
        "FP Free List",
        "Free List",  # assuming this is only integer free list?
        "Instruction Scheduler",
        "Instruction Window",  # assuming this is only integer instruction window?
        "FP Instruction Window",
        "Integer RF",
        "Integer ALU",
        "Load Store Unit",
        "LoadQ",
        "StoreQ",
        "Itlb",
    ]:
        stats = mcpat_get_unit_stats(unit, text)
        unit_stats[unit] = stats

    # TODO: make this better
    unit_power = lambda key: (
        unit_stats[key]["Runtime Dynamic"]
        if isinstance(key, str)
        else sum(unit_stats[i]["Runtime Dynamic"] for i in key)
    )

    # Now map unit stats to hotspot
    # Assuming Alpha EV6 floorplan
    # TODO: Alpha EV6 seems to be different to the Alpha21364.xml we use as our baseline?
    # Note we don't have a direct mapping, so we take some averages
    # TODO: mathematical correctness of averaging? shouldn't we consider it to just be one component with less heat spread?
    # TODO: unsure about which McPAT components to use
    hotspot_mapping = {
        "L2_left": unit_power("L2") / 3.0,
        "L2": unit_power("L2") / 3.0,
        "L2_right": unit_power("L2") / 3.0,
        "Icache": unit_power("Instruction Cache"),
        "Dcache": unit_power("Data Cache"),
        "Bpred_0": unit_power("Branch Predictor") / 3.0,
        "Bpred_1": unit_power("Branch Predictor") / 3.0,
        "Bpred_2": unit_power("Branch Predictor") / 3.0,
        "DTB_0": unit_power("FP Front End RAT") / 3.0,
        "DTB_1": unit_power("Dtlb") / 3.0,
        "DTB_2": unit_power("Dtlb") / 3.0,
        # NOTE: averaging these ones over the Add and Mul units
        "FPAdd_0": unit_power("Floating Point Units") / 4.0,
        "FPAdd_1": unit_power("Floating Point Units") / 4.0,
        "FPReg_0": unit_power("Floating Point RF") / 4.0,
        "FPReg_1": unit_power("Floating Point RF") / 4.0,
        "FPReg_2": unit_power("Floating Point RF") / 4.0,
        "FPReg_3": unit_power("Floating Point RF") / 4.0,
        "FPMul_0": unit_power("Floating Point Units") / 4.0,
        "FPMul_1": unit_power("Floating Point Units") / 4.0,
        "FPMap_0": unit_power(("FP Front End RAT", "FP Retire RAT", "FP Free List"))
        / 2.0,
        "FPMap_1": unit_power(("FP Front End RAT", "FP Retire RAT", "FP Free List"))
        / 2.0,
        "IntMap": unit_power(("Int Front End Rat", "Int Retire RAT", "Free List")),
        "IntQ": unit_power("Instruction Window"),
        "IntReg_0": unit_power("Integer RF") / 2.0,
        "IntReg_1": unit_power("Integer RF") / 2.0,
        "IntExec": unit_power("Integer ALU"),
        "FPQ": unit_power("FP Instruction Window"),
        "LdStQ": unit_power(("LoadQ", "StoreQ")),
        "ITB_0": unit_power("Itlb") / 2.0,
        "ITB_1": unit_power("Itlb") / 2.0,
    }

    return hotspot_mapping


def create_ptrace(per_unit_power: dict, previous_unit_power: dict):
    pass


def main():
    # TODO: assume we have some CFG data with branch probability info, and block IDs that map exactly
    #   to those in the DVS calling points we were given
    # TODO: current input files just give us IDs in some arbitrary order
    #   - need some format like _pathIndex_blockID_dvsCallingPointIndex
    #   - then we can easily extract per-block ID branches, and determine what should be
    #       1. all possible previous ptraces (temps)
    #       2. either weighted average, take worst-case, take most likely of these ptraces
    #     and use that as our initial temperatures
    unit_stats = mcpat_to_dict("../mcpat_out/gemm/gemm_idx0000_low.txt")
    print(unit_stats)


if __name__ == "__main__":
    main()
