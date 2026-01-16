import csv
import argparse
import xml.etree.ElementTree as ET
import re

# TODO: proper utilities python file
import write_xml

def get_stats(crit_path_stats, gm5_stats):
    print(crit_path_stats, gm5_stats)

    static_analysis = write_xml.RequiredStats()
    gm5_analysis = write_xml.RequiredStats()

    # Module index 2 (should be the PGO run)
    static_analysis.load_sum_csv_mbb_stats(crit_path_stats, 2)
    gm5_analysis.load_gem5_stats(gm5_stats)

    pe = lambda t, p: (t - p) / t * 100.0

    print(f"Gem5 {gm5_analysis.cycle_count=} and LLVM {static_analysis.cycle_count=}")
    print(f"Gem5 {gm5_analysis.total_instructions=} and LLVM {static_analysis.total_instructions=}")
    print(f"Gem5 {gm5_analysis.load_instructions=} and LLVM {static_analysis.load_instructions=}")
    print(f"Gem5 {gm5_analysis.store_instructions=} and LLVM {static_analysis.store_instructions=}")

    # compare core metrics
    # we only have one reading, so we just use percentage error
    cycle_pe = pe(gm5_analysis.cycle_count, static_analysis.cycle_count)
    instr_pe = pe(gm5_analysis.total_instructions, static_analysis.total_instructions)
    load_instr_pe = pe(gm5_analysis.load_instructions, static_analysis.load_instructions)
    store_instr_pe = pe(gm5_analysis.store_instructions, static_analysis.store_instructions)

    print(f"Cycle PE: {cycle_pe:.2f}%, Instr: {instr_pe:.2f}%, Loads: {load_instr_pe:.2f}%, Stores: {store_instr_pe:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create xml file based on Alpha given some data.")
    parser.add_argument("crit_path_stats", help="Path to MBB_stats.csv")
    parser.add_argument("gm5_stats", help="Path to stats.txt of the gem5 output")
    args = parser.parse_args()

    get_stats(args.crit_path_stats, args.gm5_stats)

