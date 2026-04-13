import argparse
import utils
import pandas as pd
import os
import math

STAT_EXPLAINED = """
# All stats are given as percentage increases
# - EDP, lower is better
# - Energy, lower is better
# - IPS, higher is better
# EDP@Constant, Energy@Constant - Measures difference between
#   ETC:        Voltage: TEI-aware,  Frequency: Baseline, 3GHz
#   Baseline:   Voltage: Baseline,   Frequency: Baseline, 3GHz
# EDP@Potential, Energy@Potential- Measures difference between
#   ETC:        Voltage: TEI-aware,  Frequency: TEI-aware
#   Baseline:   Voltage: Baseline,   Frequency: Baseline, 3GHz
# IPS@Conservative - Measures difference between
#   ETC:        Frequency: Minimum between TEI-aware, and baseline 
#   Baseline:   Frequency: Baseline, 3GHz 
# IPS@Potential - Measures difference between
#   ETC:        Frequency: TEI-aware (potential maximum)
#   Baseline:   Frequency: Baseline, 3GHz
# MaxFreq - Maximum frequency ever used in TEI-aware approach
# AverageFreq - Average frequency used, weighted by execution time
# MaxTemp - Maximum temperature ever used in TEI-aware approach
# AverageTemp - Average temperature used, weighted by execution time
"""


def percent_increase(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def add_execution_time(frequency_per_block: dict, stats_df: pd.DataFrame):
    stats_df["frequency_per_block"] = stats_df["block_id"].map(frequency_per_block)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"].astype(float)
    stats_df["frequency_per_block"] = stats_df["frequency_per_block"] * 1_000_000_000.0
    stats_df = stats_df.dropna().copy()
    stats_df["cycle_count"] = stats_df["cycle_count"].astype(float)
    stats_df["execution_time"] = (
        stats_df["cycle_count"] / stats_df["frequency_per_block"]
    )

    return stats_df


def add_energy_column(
    power_per_block: dict, frequency_per_block: dict, stats_df: pd.DataFrame
):
    stats_df = add_execution_time(frequency_per_block, stats_df)
    stats_df["core_power"] = stats_df["block_id"].map(power_per_block)
    stats_df["energy"] = stats_df["execution_time"] * stats_df["core_power"]
    stats_df = stats_df.dropna().copy()

    return stats_df


def calculate_energy(
    power_per_block: dict, frequency_per_block: dict, stats_df: pd.DataFrame
) -> float:
    stats_df = add_energy_column(power_per_block, frequency_per_block, stats_df.copy())

    return stats_df["energy"].sum()


def calculate_average_frequency(
    frequency_per_block: dict, stats_df: pd.DataFrame
) -> float:
    stats_df = add_execution_time(frequency_per_block, stats_df)
    avg_freq = (
        stats_df["frequency_per_block"] * stats_df["execution_time"]
    ).sum() / stats_df["execution_time"].sum()

    return avg_freq


def calculate_edp(
    power_per_block: dict,
    frequency_per_block: dict,
    stats_df: pd.DataFrame,
) -> float:
    stats_df = add_energy_column(power_per_block, frequency_per_block, stats_df)
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


def request_execution_get_power(
    request_spec: utils.PowerTraceRequestSpec,
    block_id: int,
    voltage: float,
    frequency: float,
) -> float:
    request_spec.change_to_other_config(block_id, voltage, frequency)
    power_trace = utils.request_power_for_specification(request_spec)
    core_power = utils.get_static_dynamic_power(power_trace, ["Core"])

    return core_power


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
        "--tei_vf_levels",
        type=str,
        default="VoltageFrequency.csv",
        help="Voltage frequency pairsp er basic block",
    )
    parser.add_argument(
        "--baseline_heat",
        type=str,
        help="Heat data file for the baseline temperatures",
    )
    parser.add_argument(
        "--heat_data",
        type=str,
        help="Heat data for ETC approach",
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
    new_vf_pairs = utils.load_voltage_frequency(args.tei_vf_levels)

    # correlate power to every BB by reading McPAT output stats
    num_blocks = stats_df["block_id"].max() + 1
    core_power_tvtf = {}
    core_power_tvcf = {}
    core_power_cvcf = {}
    core_freq_tf = {}
    core_freq_cf = {}

    request_spec = utils.PowerTraceRequestSpec(
        0,
        0.0,
        utils.load_voltage_levels_from_cfg(configs),
        args.mcpat_ins,
        args.mcpat_outs,
        args.file_prefix,
        stats_df,
        args.input_xml,
        0.0,
        configs,
    )

    utils.info(f"Number of voltage levels: {request_spec.voltage_levels}")

    for i in range(num_blocks):
        utils.info(f"Evaluating efficiency for block {i=}")
        row_data = new_vf_pairs.loc[new_vf_pairs["block_id"] == i]

        if len(row_data) == 0:
            utils.warn(
                f"Missing block, or too many voltage levels for block {i}, found {len(row_data)} entries instead of 1"
            )

            continue

        block_voltage = row_data["voltage"].iloc[0]
        block_frequency = row_data["frequency"].iloc[0]

        if math.isnan(block_frequency):
            utils.error(
                f"Got NaN block frequency for block {i}, replacing with {frequency}"
            )
            block_frequency = frequency

        utils.info(
            f"Block voltage: {block_voltage}, Block frequency: {block_frequency}"
        )

        baseline_voltage = float(
            configs[utils.MCPAT_CFG_MODULE_NAME][utils.MCPAT_BASELINE_VOLTAGE]
        )

        # TODO: instead of frequency, get the maximum frequency we would get at baseline voltage given temperature
        baseline_frequency = (
            float(configs[utils.MCPAT_CFG_MODULE_NAME][utils.MCPAT_CLOCK_RATE_MHZ])
            / 1000.0
        )

        # TEI-aware voltage and frequency
        power_tvtf = request_execution_get_power(
            request_spec, i, block_voltage, block_frequency
        )
        # TEI-aware voltage and constant frequency
        power_tvcf = request_execution_get_power(
            request_spec, i, block_voltage, baseline_frequency
        )
        # Baseline voltage and constant (baseline) frequency
        power_cvcf = request_execution_get_power(
            request_spec, i, baseline_voltage, baseline_frequency
        )

        core_power_tvtf[i] = power_tvtf
        core_power_tvcf[i] = power_tvcf
        core_power_cvcf[i] = power_cvcf

        core_freq_tf[i] = block_frequency
        core_freq_cf[i] = baseline_frequency

    # Take per-block minimum of constant frequencies, and our tei frequencies
    minimum_frequency_per_block = {}

    for i in range(num_blocks):
        maximum_frequency = core_freq_tf.get(i, frequency)
        minimum_frequency_per_block[i] = min(maximum_frequency, frequency)

    utils.info(
        f"Conservative frequency sum difference: {sum(minimum_frequency_per_block[k] - frequency for k in minimum_frequency_per_block.keys())}"
    )
    # Peak temperature (celsius, value):
    #   TEI voltage, Constant frequency : Baseline voltage, Constant frequency
    # TODO: Additional data required: temperature values assuming baseline voltage, and constant frequency
    # TODO: make easy to parse for dataframe

    edp_constant_etc = calculate_edp(core_power_tvcf, core_freq_cf, stats_df.copy())
    edp_potential_etc = calculate_edp(core_power_tvtf, core_freq_tf, stats_df.copy())
    edp_baseline = calculate_edp(core_power_cvcf, core_freq_cf, stats_df.copy())

    print(f"{edp_constant_etc=}, {edp_baseline=}, {edp_potential_etc=}")

    ips_potential_etc = calculate_ips(core_freq_tf, stats_df.copy())
    ips_conservative_etc = calculate_ips(minimum_frequency_per_block, stats_df.copy())
    ips_baseline = calculate_ips(core_freq_cf, stats_df.copy())

    utils.info(f"{ips_conservative_etc=} {ips_baseline=}")

    energy_constant_etc = calculate_energy(
        core_power_tvcf, core_freq_cf, stats_df.copy()
    )
    energy_potential_etc = calculate_energy(
        core_power_tvtf, core_freq_tf, stats_df.copy()
    )
    energy_baseline = calculate_energy(core_power_cvcf, core_freq_cf, stats_df.copy())

    energy_constant_increase = percent_increase(energy_constant_etc, energy_baseline)
    energy_potential_increase = percent_increase(energy_potential_etc, energy_baseline)

    edp_constant_increase = percent_increase(edp_constant_etc, edp_baseline)
    edp_potential_increase = percent_increase(edp_potential_etc, edp_baseline)

    ips_conservative_increase = percent_increase(ips_conservative_etc, ips_baseline)
    ips_potential_increase = percent_increase(ips_potential_etc, ips_baseline)

    average_frequency = (
        calculate_average_frequency(core_freq_tf, stats_df.copy()) / 1.0e9
    )
    max_frequency = max(core_freq_tf.values())

    etc_heat = utils.load_program_heats(args.heat_data)
    baseline_heat = utils.load_program_heats(args.baseline_heat)

    heat_df = etc_heat.merge(
        baseline_heat, on="block_id", how="outer", suffixes=("_etc", "_baseline")
    )
    heat_df = heat_df.merge(
        stats_df[["block_id", "cycle_count"]].copy(), on="block_id", how="inner"
    )

    heat_df["temp_diff"] = heat_df["temp_max_etc"] - heat_df["temp_max_baseline"]
    # Converting to celsius
    heat_df["temp_max_etc"] = heat_df["temp_max_etc"] - 273.15
    heat_df["temp_max_baseline"] = heat_df["temp_max_baseline"] - 273.15
    heat_df = heat_df[
        ["block_id", "temp_max_etc", "temp_max_baseline", "temp_diff", "cycle_count"]
    ].copy()

    # use TEI-aware frequencies
    heat_df = add_execution_time(core_freq_tf, heat_df)

    maximum_temperature = heat_df["temp_max_etc"].max()
    average_temperature = (
        heat_df["temp_max_etc"] * heat_df["execution_time"]
    ).sum() / heat_df["execution_time"].sum()

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
        f.write(f"AverageFreq: {average_frequency:.2f}\n")
        f.write(f"MaxFreq: {max_frequency:.2f}\n")
        f.write(f"AverageTemp: {average_temperature:.2f}\n")
        f.write(f"MaxTemp: {maximum_temperature:.2f}\n")


if __name__ == "__main__":
    main()
