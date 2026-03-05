import argparse
import utils
import pandas as pd


def calculate_edp(
    power_per_block: dict,
    frequency_per_block: dict,
    stats_df: pd.DataFrame,
) -> float:
    # Now have: execution time per basic block, power of core per basic block, should have committed instructions per basic block too
    # TODO: use frequency per block
    stats_df["frequency_per_block"] = stats_df["block_id"].map(frequency_per_block)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"].astype(float)
    stats_df["frequency_per_block"] = stats_df["block_id"] * 1_000_000_000.0
    stats_df["cycle_count"] = stats_df["cycle_count"].astype(float)
    stats_df["execution_time"] = (
        stats_df["cycle_count"] / stats_df["frequency_per_block"]
    )
    stats_df["core_power"] = stats_df["block_id"].map(power_per_block)
    stats_df["energy"] = stats_df["execution_time"] * stats_df["core_power"]
    stats_df = stats_df.dropna()
    stats_df["ipc"] = stats_df["instr_count"] / stats_df["cycle_count"]
    # Per-block edp
    stats_df["edp"] = stats_df["energy"] * stats_df["execution_time"]
    # Overall edp, PT^2, we calculate energy per block, take sum, and then multiply by total time (delay)
    overall_edp = stats_df["energy"].sum() * stats_df["execution_time"].sum()

    print("Total energy:", stats_df["energy"].sum())

    print(f"Estimating overall EDP to be: {overall_edp}")

    return overall_edp


def calculate_ips(frequency_per_block: dict, stats_df: pd.DataFrame):
    # TODO: instead of execution time, we need to calculate frequency per block given the voltage
    # we're running at (and the predicted initial temperature, so we need extra data not necessary available)
    stats_df["frequency_per_block"] = stats_df["block_id"].map(frequency_per_block)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"].astype(float)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"] * 1_000_000_000.0
    stats_df["cycle_count"] = stats_df["cycle_count"].astype(float)
    stats_df["true_execution_time"] = (
        stats_df["cycle_count"] / stats_df["frequency_per_block"]
    )
    overall_ips = stats_df["instr_count"].sum() / stats_df["true_execution_time"].sum()

    return overall_ips


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
    # MHz -> GHz
    frequency = (
        float(configs[utils.MCPAT_CFG_MODULE_NAME][utils.MCPAT_CLOCK_RATE_MHZ]) / 1000.0
    )
    new_voltage_levels = utils.load_voltage_levels(args.new_voltage_levels)

    # TODO: too much precision loss is possible?
    # TODO: deal with any N/A results (although shouldn't be any)
    stats_df["execution_time"] = stats_df["cycle_count"].astype(float) / frequency
    total_execution_time = stats_df["execution_time"].sum()

    # correlate power to every BB by reading McPAT output stats
    num_blocks = stats_df["block_id"].max() + 1
    core_power_per_block = {}
    core_power_per_baseline_block = {}
    core_frequency_per_block = {}
    core_frequency_per_baseline_block = {}
    core_frequency_constant = {}
    request_spec = utils.PowerTraceRequestSpec(
        0,
        0.0,
        utils.load_voltage_levels_from_cfg(configs),
        args.mcpat_ins,
        args.mcpat_outs,
        args.file_prefix,
        stats_df,
        args.input_xml,
        configs,
    )

    for i in range(num_blocks):
        print(f"Evaluating efficiency for block {i=}")
        block_voltage = new_voltage_levels.loc[new_voltage_levels["block_id"] == i]
        core_frequency_constant[i] = frequency

        if len(block_voltage) != 1:
            utils.warn(
                f"Missing block, or too many voltage levels for block {i}, found {len(block_voltage)} entries instead of 1"
            )

            continue

        desired_voltage = block_voltage.iloc[0]["voltage_level"]
        baseline_voltage = float(configs["mcpat"]["BASELINE_VOLTAGE_LEVEL"])

        desired_frequency = block_voltage.iloc[0]["obtained_frequency"]
        # TODO: instead of frequency, get the maximum frequency we would get at baseline voltage given temperature
        baseline_frequency = (
            float(configs[utils.MCPAT_CFG_MODULE_NAME][utils.MCPAT_CLOCK_RATE_MHZ])
            / 1000.0
        )

        request_spec.change_to_other_config(i, desired_voltage)
        # TODO: we should expect this to exist, but sometimes it doesn't and it makes things very complicated
        power_trace = utils.request_power_for_specification(request_spec)
        core_power = utils.get_static_dynamic_power(power_trace, ["Core"])
        core_power_per_block[i] = core_power
        core_frequency_per_block[i] = desired_frequency

        request_spec.change_to_other_config(i, baseline_voltage)
        baseline_power_trace = utils.request_power_for_specification(request_spec)
        core_baseline_power = utils.get_static_dynamic_power(
            baseline_power_trace, ["Core"]
        )
        core_power_per_baseline_block[i] = core_baseline_power
        core_frequency_per_baseline_block[i] = baseline_frequency

    edp_tei_voltage_constant_freq = calculate_edp(
        core_power_per_block, core_frequency_constant, stats_df.copy()
    )
    edp_baseline_voltage_constant_freq = calculate_edp(
        core_power_per_baseline_block,
        core_frequency_constant,
        stats_df.copy(),
    )

    ips_tei_voltage_tei_freq = calculate_ips(core_frequency_per_block, stats_df.copy())
    ips_tei_voltage_constant_freq = calculate_ips(
        core_frequency_constant, stats_df.copy()
    )
    # TODO: additional one we need is running the tei voltages at new voltage, and maximum allowed freuqency at those new temperatures/voltages

    percent_down = (
        (edp_baseline_voltage_constant_freq - edp_tei_voltage_constant_freq)
        / edp_baseline_voltage_constant_freq
    ) * 100.0
    ips_up = (
        (ips_tei_voltage_tei_freq - ips_tei_voltage_constant_freq)
        / ips_tei_voltage_constant_freq
    ) * 100.0

    print(f"EDP percentage improvement: {percent_down:.4f}%")
    print(f"IPS improvement: {ips_up:.4f}%")

    # Explanation of results
    # We compare our energy-delay tradeoff tbetween our tie-aware voltages, and the baseline voltage, assuming we run both at 3GHz
    # We compute the IPS percentage improvement for two scenarios
    # - assuming both approaches are run with maximum frequency according to TEI
    # - assuming we run our approach at maximum tei-aware freuency, compared to baseline voltage at baseline frequency
    # - assuming run our approach at constant frequency, compared to baseline voltage at maximum tei frequency
    # TODO: best comparison would be both approaches running at TEI-aware frequency?
    #  including calculating our EDP at that tei-aware frequency
    with open("efficiencyStats.txt", "a") as f:
        f.write(f"Test name: {args.file_prefix}\n")
        f.write(
            f"EDP percentage improvement (assume constant frequency): {percent_down:.4f}%\n"
        )
        # TODO: need some additional things t cmpute
        # f.write(
        #     f"IPS percentage improvement (assume max-allowed tei frequency): {ips_up:.4f}%\n"
        # )
        f.write(
            f"IPS percentage improvement (assume max-allowed tei frequency for ETC, compared to constant frequency): {ips_up:.4f}%\n"
        )


if __name__ == "__main__":
    main()
