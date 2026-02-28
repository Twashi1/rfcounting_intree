import argparse
import utils


def calculate_edp(
    power_per_block: dict, frequency: float, stats_df: pd.DataFrame
) -> float:
    # Now have: execution time per basic block, power of core per basic block, should have committed instructions per basic block too
    stats_df["execution_time"] = stats_df["cycle_count"].astype(float) / frequency
    stats_df["core_power"] = stats_df["block_id"].map(power_per_block)
    stats_df["energy"] = stats_df["execution_time"] * stats_df["core_power"]
    stats_df = stats_df.dropna()
    stats_df["ipc"] = stats_df["instr_count"] / stats_df["cycle_count"]
    # Per-block edp
    stats_df["edp"] = stats_df["energy"] * stats_df["execution_time"]
    # Overall edp, PT^2, we calculate energy per block, take sum, and then multiply by total time (delay)
    overall_edp = stats_df["energy"].sum() * stats_df["execution_time"].sum()

    # TODO: instead of execution time, we need to calculate frequency per block given the voltage
    # we're running at (and the predicted initial temperature, so we need extra data not necessary available)
    overall_ips = stats_df["instr_count"].sum() / stats_df["execution_time"].sum()

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
    # TODO: utils some function to request power trace only if exists, and error otherwise
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
        "--mcpat_ins",
        type=str,
        help="Directory with mcpat input stats per path/basic block",
    )
    parser.add_argument(
        "--input_xml",
        type=str,
        help="Default file to base our McPAT inputs off",
        default="./mcpat_inputs/Alpha21364.xml",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        help="Last level directory name in mcpat outputs, usually the program name gemm/2mm/atax/covariance",
    )
    parser.add_argument(
        "--new_voltage_levels",
        type=str,
        default="VoltageLevels.csv",
        help="The voltage levels to assume are being used for each basic block when estimating power",
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

    # TODO: too much precision loss is possible?
    # TODO: deal with any N/A results (although shouldn't be any)
    stats_df["execution_time"] = stats_df["cycle_count"].astype(float) / frequency
    total_execution_time = stats_df["execution_time"].sum()

    # correlate power to every BB by reading McPAT output stats
    num_blocks = stats_df["block_id"].max() + 1
    core_power_per_block = {}
    core_power_per_baseline_block = {}
    request_spec = utils.PowerTraceRequestSpec(
        0,
        0.0,
        utils.load_voltage_levels_from_cfg(configs),
        "",
        "",
        args.file_prefix,
        stats_df,
        "",
        configs,
    )

    for i in range(num_blocks):
        block_voltage = new_voltage_levels.loc[new_voltage_levels["block_id" == i]]

        if len(block_voltage) != 1:
            raise ValueError(
                f"Missing block, or too many voltage levels for block {i}, found {len(block_voltage)} entries instead of 1"
            )

        desired_voltage = block_voltage.iloc[0]
        request_spec.change_to_other_config(i, desired_voltage)
        baseline_voltage = configs["mcpat"]["BASELINE_VOLTAGE_LEVEL"]

        power_trace = utils.request_power_for_specification(
            request_spec, expect_exists=True
        )
        # request_power_for_specification_must_exist
        core_power_per_block[i] = power_trace["Core"]

        request_spec.change_to_other_config(i, block_voltage)
        baseline_power_trace = utils.request_power_for_specification(request_spec)

        core_power_per_baseline_block[i] = baseline_power_trace[i]

    new_edp = calculate_edp(core_power_per_block, frequency, stats_df)
    old_edp = calculate_edp(core_power_per_baseline_block, frequency, stats_df)

    percent_change = ((new_edp - old_edp) / old_edp) * 100.0

    print(f"EDP percentage improvement: {percent_change:.4f}%")


if __name__ == "__main__":
    main()
