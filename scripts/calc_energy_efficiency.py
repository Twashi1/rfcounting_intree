import argparse
import utils


def calculate_edp(mcpat_df, frequency, stats_df) -> float:
    # TODO: apply filter to mcpat_df given voltage levels
    merged = mcpat_df.merge(stats_df, on="block_id")
    merged = merged.dropna(axis=0, how="any")

    # # Now have: execution time per basic block, power of core per basic block, should have committed instructions per basic block too
    # merged["energy"] = merged["execution_time"] * merged["Core"]
    # merged["ipc"] = merged["instr_count"] / merged["cycle_count"]
    # # TODO: maybe energy * ipc?
    # # TODO: might need to check for IPC = 0 (would suggest instructions are 0 which is exceptional case, but still)
    # # edp = P * (instr / (IPC * f))
    # merged["edp"] = (merged["energy"] * merged["instr_count"]) / (
    #     merged["ipc"] * frequency
    # )

    # Now have: execution time per basic block, power of core per basic block, should have committed instructions per basic block too
    merged["energy"] = merged["execution_time"] * merged["Core"]
    merged["ipc"] = merged["instr_count"] / merged["cycle_count"]
    # Per-block edp
    merged["edp"] = merged["energy"] * merged["execution_time"]
    # Overall edp, PT^2, we calculate energy per block, take sum, and then multiply by total time (delay)
    overall_edp = merged["energy"].sum() * merged["execution_time"].sum()

    # print(
    #     merged[
    #         [
    #             "block_id",
    #             "energy",
    #             "ipc",
    #             "instr_count",
    #             "execution_time",
    #             "cycle_count",
    #             "Core",
    #             "edp",
    #         ]
    #     ]
    # )

    # TODO: this average instructions per cycle is not correct
    # overall_edp = merged["energy"].sum() * (
    #     merged["instr_count"].sum() / (merged["ipc"].mean() * frequency)
    # )

    print(f"Estimating overall EDP to be: {overall_edp}")

    return overall_edp


def main():
    parser = argparse.ArgumentParser(
        description="Calculate EDP per block, and over the whole program"
    )
    parser.add_argument(
        "--stats",
        help="Path to some _STD.csv file, expecting stats per machine basic block",
    )
    parser.add_argument(
        "--input_cfg",
        type=str,
        default="scripts/configs.cfg",
        help="General config file",
    )
    parser.add_argument(
        "--mcpat_outs",
        type=str,
        help="Directory with mcpat output power per path/basic block",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="",
        help="Last level directory name in mcpat outputs, usually the program name gemm/2mm/atax/covariance, can leave default to assume based on mcpat_outs",
    )
    parser.add_argument(
        "--new_voltage_levels",
        type=str,
        default="VoltageLevels.csv",
        help="The voltage levels to assume are being used for each basic block when estimating power",
    )
    parser.add_argument(
        "--old_voltage_levels",
        type=str,
        default="VoltageLevels.csv",
        help="The old voltage levels/fixed voltage levels to compare to",
    )
    args = parser.parse_args()

    # STEPS
    # - for each block, get execution **time**
    # - for each block, calculate both IPC and energy (power * time)
    # - over the entire program, sum up the energy calculation, and take a whole-program average of IPC
    stats_df = utils.load_standard_stat_file(args.stats)
    configs = utils.load_cfg(args.input_cfg)
    # MHz -> Hz
    frequency = float(configs["mcpat"]["CLOCK_RATE"]) * 1_000_000.0
    new_voltage_levels = utils.load_voltage_levels(args.new_voltage_levels)
    old_voltage_levels = utils.load_voltage_levels(args.old_voltage_levels)

    print(new_voltage_levels, old_voltage_levels)

    # TODO: too much precision loss is possible?
    # TODO: deal with any N/A results (although shouldn't be any)
    stats_df["execution_time"] = stats_df["cycle_count"].astype(float) / frequency
    total_execution_time = stats_df["execution_time"].sum()

    # correlate power to every BB by reading McPAT output stats
    # TODO: apply some extra name on most columns to be "Core_power" and "L2_power" etc.
    mcpat_df = utils.load_folder_mcpat(args.mcpat_outs, args.file_prefix)
    # Filter voltages
    new_mcpat_df = mcpat_df.loc[
        mcpat_df.set_index(["block_id", "voltage"]).index.isin(
            new_voltage_levels.set_index(["block_id", "voltage_level"]).index
        )
    ]
    old_mcpat_df = mcpat_df.loc[
        mcpat_df.set_index(["block_id", "voltage"]).index.isin(
            old_voltage_levels.set_index(["block_id", "voltage_level"]).index
        )
    ]

    new_edp = calculate_edp(new_mcpat_df, frequency, stats_df)
    old_edp = calculate_edp(old_mcpat_df, frequency, stats_df)

    percent_change = ((new_edp - old_edp) / old_edp) * 100.0

    print(f"EDP percentage improvement: {percent_change:.4f}%")


if __name__ == "__main__":
    main()
