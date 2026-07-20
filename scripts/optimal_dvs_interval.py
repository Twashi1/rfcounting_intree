import utils
import pandas as pd
import mcpat_to_ptrace as mcpt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from pathlib import Path


def clear_cached():
    mcinputs = Path(f"./mcpat_inputs/gemm_10k_instr/")
    mcoutputs = Path(f"./mcpat_out/gemm_10k_instr/")

    if mcinputs.exists():
        shutil.rmtree(mcinputs)

    if mcoutputs.exists():
        shutil.rmtree(mcoutputs)


def calculate_data():
    stat_files = "./gem5_stats/gemm_10k_instr/"

    config = utils.load_cfg("./scripts/configs.cfg")

    flp_df = pd.read_csv(
        "./hotspot_files/ev6.flp",
        delim_whitespace=True,
        header=None,
        index_col=0,
        comment="#",
    )

    flp_df.columns = ["width", "height", "leftx", "bottomy"]
    flp_df.index.name = "unit"
    flp_df = flp_df.reset_index()
    flp_df["width"] = flp_df["width"].astype(float)
    flp_df["height"] = flp_df["height"].astype(float)
    flp_df["leftx"] = flp_df["leftx"].astype(float)
    flp_df["bottomy"] = flp_df["bottomy"].astype(float)
    flp_df["area"] = flp_df["width"] * flp_df["height"]

    all_temps = []
    max_temps = []
    # area_weighted_temps = []
    area_weighted_no_cache = []
    instr_counts = []
    selected_voltages = []
    selected_frequencies = []
    last_temperature = 350  # default to 77 celsius

    for i in range(995):
        file_name = stat_files + f"stats_{i:05d}.txt"

        utils.create_standard_stat_file(file_name, f"./stats/gemm_10k_instr")
        gem5_stats = utils.load_standard_stat_file(f"./stats/gemm_10k_instr_STD.csv")

        gem5_stats["block_id"] = i

        vf_pairs = utils.tei_select_vf_pairs(
            config,
            last_temperature - 273.15,
            [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
            False,
            False,
            True,
        )

        gem5_cycle_count = gem5_stats[utils.BUSY_CYCLES].sum()

        best_edp = float("inf")
        best_vf_pair = None
        best_power_trace = None

        for voltage, frequency in vf_pairs:
            gem5_trace_request = utils.PowerTraceRequestSpec(
                i,
                voltage,
                [0.6, 0.65, 0.7, 0.75, 0.8],
                f"./mcpat_inputs/gemm_10k_instr",
                f"./mcpat_out/gemm_10k_instr",
                f"gemm_10k_instr",
                gem5_stats,
                "./mcpat_inputs/Alpha21364.xml",
                frequency,
                config,
            )

            gem5_power = utils.request_power_for_specification(gem5_trace_request)
            assert gem5_power is not None

            if "Core" not in gem5_power or gem5_power["Core"] is None:
                print("[WARN] Skipping one configuration")
                continue

            core_power_gem5 = utils.get_static_dynamic_power(gem5_power, ["Core"])

            # calculate rough EDP
            execution_time = gem5_cycle_count / (frequency * 1.0e9)
            energy = core_power_gem5 * execution_time
            edp = energy * execution_time

            if edp < best_edp:
                best_edp = edp
                best_vf_pair = (voltage, frequency)
                best_power_trace = gem5_power

        assert best_power_trace is not None
        assert best_vf_pair is not None

        best_voltage, best_frequency = best_vf_pair

        selected_voltages.append(best_voltage)
        selected_frequencies.append(best_frequency)

        hotspot_ptrace_gem5 = utils.mcpat_to_hotspot_units(
            best_power_trace, flp_df, True
        )

        final_heat_gem5 = mcpt.get_hotspot_temp(
            hotspot_ptrace_gem5,
            float(gem5_cycle_count) / float(best_frequency * 1.0e9),
            config,
            "./hotspot_files/example.config",
            None,
            2,
            best_frequency,
            f"{i:04d}_{best_frequency:.1f}Hz",
        )

        # TODO: take max temp or take area-weighted average temp
        max_temp_gem5 = max(final_heat_gem5.values())
        avg_temp_gem5 = sum(final_heat_gem5.values()) / len(final_heat_gem5)

        execution_units = [
            "FPAdd_0",
            "FPAdd_1",
            "FPReg_0",
            "FPReg_1",
            "FPReg_2",
            "FPReg_3",
            "FPMul_0",
            "FPMul_1",
            "FPQ",
            "IntQ",
            "IntExec",
            "IntReg_0",
            "IntReg_1",
        ]

        execution_total_area = flp_df.loc[
            flp_df["unit"].isin(execution_units),
            "area",
        ].sum()

        # Should only be one entry, max is arbitrary
        area_fraction = (
            lambda unit_name, area: flp_df.loc[
                flp_df["unit"] == unit_name, "area"
            ].max()
            / area
        )

        area_weighted_temp_no_cache = 0.0

        for unit_name in execution_units:
            temp = final_heat_gem5[unit_name]
            area_weighted_temp_no_cache += temp * area_fraction(
                unit_name, execution_total_area
            )

        print(f"Temp: {area_weighted_temp_no_cache}K")
        last_temperature = area_weighted_temp_no_cache

        area_weighted_no_cache.append(area_weighted_temp_no_cache)

        all_temps.append(avg_temp_gem5)
        max_temps.append(max_temp_gem5)
        # TODO: real instruction count ends up being ~14k, but read from file is even better
        instr_counts.append(14_000 * i)

        print("Done!")

    df = pd.DataFrame(
        {
            "instructions": instr_counts,
            "temperature_k": all_temps,
            "temperature_weighted_k": area_weighted_no_cache,
            "voltages": selected_voltages,
            "frequencies": selected_frequencies,
        }
    )

    df.to_pickle("temperatures1.pkl")


def analysis():
    df = pd.read_pickle("temperatures.pkl")

    sns.set_theme(style="whitegrid")

    ax = sns.lineplot(
        data=df,
        x="instructions",
        y="temperature_weighted_k",
        marker="o",
    )

    ax.set_xlabel("Instructions")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Temperature vs. Instructions")

    plt.tight_layout()
    plt.show()


def analysis_with_bar():
    df = pd.read_pickle("temperatures.pkl")

    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Temperature line
    sns.lineplot(
        data=df,
        x="instructions",
        y="temperature_weighted_k",
        marker="o",
        ax=ax1,
        color="tab:red",
    )

    ax1.set_xlabel("Instructions")
    ax1.set_ylabel("Temperature (K)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Secondary axis for voltage
    ax2 = ax1.twinx()

    ax2.scatter(
        df["instructions"],
        df["voltages"],
        color="tab:blue",
        s=20,
        marker="o",
        label="Voltage",
        zorder=3,
    )

    ax2.set_ylabel("Voltage (V)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Temperature and Voltage vs. Instructions")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # clear_cached()
    calculate_data()
    analysis_with_bar()
