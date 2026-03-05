import argparse

from numpy import power
import utils
import pandas as pd
import os

STAT_EXPLAINED = """
# All stats are given as percentage increases
# - EDP, lower is better
# - Energy, lower is better
# - IPS, higher is better
# EDP@Constant, Energy@Constant - Measures difference between
#   ETC:        Voltage: TEI-aware,  Frequency: Baseline, 3GHz
#   Baseline:   Voltage: Baseline,   Frequency: Baseline, 3GHz
# EDP@Potential, Energy@Potential - Measures difference between
#   ETC:        Voltage: TEI-aware, Frequency: TEI-aware (potential maximum)
#   Baseline:   Voltage: Baseline,  Frequency: Baseline, 3GHz 
# IPS@Conservative - Measures difference between
#   ETC:        Frequency: Minimum between TEI-aware, and baseline 
#   Baseline:   Frequency: Baseline, 3GHz 
# IPS@Potential - Measures difference between
#   ETC:        Frequency: TEI-aware (potential maximum)
#   Baseline:   Frequency: Baseline, 3GHz
#
# - Intended comparison is EDP@Constant, and IPS@Conservative
#   - Even if we potentially have overhead allowing us a higher frequency under the selected voltages, we intend to only apply DVS, and thus we cannot make use of enhanced frequencies
#   - IPS@Conservative models, to some degree, where we might lose performance since we were not able to support the required frequency with the selected voltage
#   - In almost all configurations, IPS@Conservative should give 0.0% performance loss, or close to 0.0% performance loss, since we round up voltages to ensure we can always support the target frequency
# - IPS not compared between ETC running at baseline frequency, and Baseline running at baseline frequency
"""


def percent_increase(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def add_energy_column(
    power_per_block: dict, frequency_per_block: dict, stats_df: pd.DataFrame
):
    stats_df["frequency_per_block"] = stats_df["block_id"].map(frequency_per_block)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"].astype(float)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"] * 1_000_000_000.0
    stats_df = stats_df.dropna().copy()
    stats_df["cycle_count"] = stats_df["cycle_count"].astype(float)
    stats_df["execution_time"] = (
        stats_df["cycle_count"] / stats_df["frequency_per_block"]
    )
    stats_df["core_power"] = stats_df["block_id"].map(power_per_block)
    stats_df["energy"] = stats_df["execution_time"] * stats_df["core_power"]
    stats_df = stats_df.dropna().copy()

    return stats_df


def calculate_energy(
    power_per_block: dict, frequency_per_block: dict, stats_df: pd.DataFrame
) -> float:
    stats_df = add_energy_column(power_per_block, frequency_per_block, stats_df.copy())

    return stats_df["energy"].sum()


def calculate_edp(
    power_per_block: dict,
    frequency_per_block: dict,
    stats_df: pd.DataFrame,
) -> float:
    stats_df = add_energy_column(power_per_block, frequency_per_block, stats_df)
    # Now have: execution time per basic block, power of core per basic block, should have committed instructions per basic block too
    stats_df["ipc"] = stats_df["instr_count"] / stats_df["cycle_count"]
    # Per-block edp
    stats_df["edp"] = stats_df["energy"] * stats_df["execution_time"]
    # Overall edp, PT^2, we calculate energy per block, take sum, and then multiply by total time (delay)
    overall_edp = stats_df["energy"].sum() * stats_df["execution_time"].sum()

    return overall_edp


def calculate_ips(frequency_per_block: dict, stats_df: pd.DataFrame):
    # TODO: instead of execution time, we need to calculate frequency per block given the voltage
    # we're running at (and the predicted initial temperature, so we need extra data not necessary available)
    stats_df["frequency_per_block"] = stats_df["block_id"].map(frequency_per_block)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"].astype(float)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"] * 1_000_000_000.0
    stats_df["cycle_count"] = stats_df["cycle_count"].astype(float)
    stats_df = stats_df.dropna().copy()

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

        desired_voltage = float(block_voltage.iloc[0]["voltage_level"])
        baseline_voltage = float(configs["mcpat"]["BASELINE_VOLTAGE_LEVEL"])

        desired_frequency = float(block_voltage.iloc[0]["obtained_frequency"])
        # TODO: instead of frequency, get the maximum frequency we would get at baseline voltage given temperature
        baseline_frequency = (
            float(configs[utils.MCPAT_CFG_MODULE_NAME][utils.MCPAT_CLOCK_RATE_MHZ])
            / 1000.0
        )

        # Calculating the power where we use the TEI-aware voltage
        request_spec.change_to_other_config(i, desired_voltage)
        # TODO: we should expect this to exist, but sometimes it doesn't and it makes things very complicated
        power_trace = utils.request_power_for_specification(request_spec)
        core_power = utils.get_static_dynamic_power(power_trace, ["Core"])
        core_power_per_block[i] = core_power
        core_frequency_per_block[i] = desired_frequency

        # Calculating the power where we use the baseline voltage
        request_spec.change_to_other_config(i, baseline_voltage)
        baseline_power_trace = utils.request_power_for_specification(request_spec)
        core_baseline_power = utils.get_static_dynamic_power(
            baseline_power_trace, ["Core"]
        )
        core_power_per_baseline_block[i] = core_baseline_power
        core_frequency_per_baseline_block[i] = baseline_frequency

    # Take per-block minimum of constant frequencies, and our tei frequencies
    # NOTE: keys should be the same, but just for safety
    minimum_frequency_per_block = {
        k: min(core_frequency_per_block[k], core_frequency_constant[k])
        for k in core_frequency_per_block.keys() & core_frequency_constant.keys()
    }

    # TODO:
    #   the comparisons to make are
    #   potentially consider taking the minimum between constant frequency, and tei frequency
    #   representing the possibility we will select too low a voltage to support our target frequency
    # EDP:
    #   TEI voltage, Constant frequency : Baseline voltage, Constant frequency
    # IPS:
    #   TEI frequency : Constant frequency
    # Energy:
    #   TEI voltage, Constant frequency : Baseline voltage, Constant frequency
    # Peak temperature (celsius, value):
    #   TEI voltage, Constant frequency : Baseline voltage, Constant frequency
    # TODO: Additional data required: temperature values assuming baseline voltage, and constant frequency
    # TODO: make easy to parse for dataframe

    edp_constant_etc = calculate_edp(
        core_power_per_block, core_frequency_constant, stats_df.copy()
    )
    edp_potential_etc = calculate_edp(
        core_power_per_block, core_frequency_per_block, stats_df.copy()
    )
    edp_baseline = calculate_edp(
        core_power_per_baseline_block,
        core_frequency_constant,
        stats_df.copy(),
    )

    ips_potential_etc = calculate_ips(core_frequency_per_block, stats_df.copy())
    ips_conservative_etc = calculate_ips(minimum_frequency_per_block, stats_df.copy())
    ips_baseline = calculate_ips(core_frequency_constant, stats_df.copy())

    energy_constant_etc = calculate_energy(
        core_power_per_block, core_frequency_constant, stats_df.copy()
    )
    energy_potential_etc = calculate_energy(
        core_power_per_block, core_frequency_per_block, stats_df.copy()
    )
    energy_baseline = calculate_energy(
        core_power_per_baseline_block,
        core_frequency_constant,
        stats_df.copy(),
    )

    energy_constant_increase = percent_increase(energy_constant_etc, energy_baseline)
    energy_potential_increase = percent_increase(energy_potential_etc, energy_baseline)

    edp_constant_increase = percent_increase(edp_constant_etc, edp_baseline)
    edp_potential_increase = percent_increase(edp_potential_etc, edp_baseline)

    ips_conservative_increase = percent_increase(ips_conservative_etc, ips_baseline)
    ips_potential_increase = percent_increase(ips_potential_etc, ips_baseline)

    stats_existed = os.path.exists("efficiencyStats.txt")

    with open("efficiencyStats.txt", "a") as f:
        if not stats_existed:
            f.write(STAT_EXPLAINED)

        f.write(f"Test name: {args.file_prefix}\n")
        f.write(f"Energy@Constant: {energy_constant_increase:.4f}%\n")
        f.write(f"Energy@Potential: {energy_potential_increase:.4f}%\n")

        f.write(f"EDP@Constant: {edp_constant_increase:.4f}%\n")
        f.write(f"EDP@Potential: {edp_potential_increase:.4f}%\n")

        f.write(f"IPS@Conservative: {ips_conservative_increase:.4f}%\n")
        f.write(f"IPS@Potential: {ips_potential_increase:.4f}%\n")


if __name__ == "__main__":
    main()
