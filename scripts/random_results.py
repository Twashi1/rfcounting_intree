# Generating random results as needed
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Can set theme here
sns.set_theme(style="whitegrid")

sfvv = utils.load_efficiency_stats("./results/sfvv_per_program.txt")
vfvv = utils.load_efficiency_stats("./results/vfvv_per_program.txt")


def load_program_data(program_names, directory="block_heats"):
    results = {}

    ESTIMATE_SWITCHING_COST = 0.5 * 1e-6  # 0.5 microseconds

    for program_name in program_names:
        df = utils.load_program_heats(f"./{directory}/{program_name}_ProgramHeat.csv")

        total_dvfs = df["dvs_calling_count"].sum()
        total_execution_time = df["execution_time"].sum()
        total_cost = total_dvfs * ESTIMATE_SWITCHING_COST
        new_time = total_execution_time + total_cost
        slowdown = (new_time - total_execution_time) / total_execution_time * 100

        results[program_name] = {
            "dvfs_calling_count": total_dvfs,
            "execution_time": total_execution_time,
            "dvfs_per_ms": total_dvfs / (total_execution_time * 1.0e3),
            "performance_slowdown%": slowdown,
        }

    return results


def get_median(a):
    return np.median(a)


def dict_to_latex_table(data, output_file, float_fmts):
    # collect all stat keys
    stats = set()
    for v in data.values():
        stats.update(v.keys())
    stats = sorted(stats)

    # begin LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l" + "r" * len(stats) + "}")
    lines.append(r"\hline")
    lines.append("Program & " + " & ".join(stats) + r" \\")
    lines.append(r"\hline")

    # rows
    for program, vals in data.items():
        row = [program]
        for i, stat in enumerate(stats):
            val = vals.get(stat, "")
            if isinstance(val, float):
                val = format(val, float_fmts[i])
            row.append(str(val))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Program Statistics}")
    lines.append(r"\end{table}")

    # write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def get_efficiency_stat_min_max(efficiency_stats, stat_name: str, flip: bool):
    stat_min = float("inf")
    stat_max = 0.0

    for program_name, results in efficiency_stats.items():
        if stat_name not in results:
            continue

        stat_value = float(results[stat_name])

        if flip:
            stat_value = -stat_value

        stat_min = min(stat_min, stat_value)
        stat_max = max(stat_max, stat_value)

    return stat_min, stat_max


def get_efficiency_stat_list(efficiency_stats, stat_name: str):
    for program_name, results in efficiency_stats.items():
        if stat_name not in results:
            continue

        stat_value = float(results[stat_name])

        yield stat_value


sf_edp_values = list(get_efficiency_stat_list(sfvv, "edp_constant"))
sf_edp_avg = sum(sf_edp_values) / len(sf_edp_values)

print(f"Average EDP: {sf_edp_avg}")

program_names = list(sfvv.keys())
sfvv_additional = load_program_data(program_names, "const_path_block_heats")
vfvv_additional = load_program_data(program_names, "var_path_block_heats")

dict_to_latex_table(
    sfvv_additional, "./results/sfvv_latex_table.txt", [".0f", ".3f", ".3f", ".2f"]
)
dict_to_latex_table(
    vfvv_additional, "./results/vfvv_latex_table.txt", [".0f", ".3f", ".3f", ".2f"]
)

dvfs_per_ms = []

for program, stats in sfvv_additional.items():
    dvfs_per_ms.append(stats["dvfs_per_ms"])

median_dvfs_per_ms = np.median(dvfs_per_ms)
print(f"Median DVFS frequency: {median_dvfs_per_ms} calls/ms")

sf_edp_min, sf_edp_max = get_efficiency_stat_min_max(sfvv, "edp_constant", True)
vf_edp_min, vf_edp_max = get_efficiency_stat_min_max(vfvv, "edp_potential", True)

sf_temp_min, sf_temp_max = get_efficiency_stat_min_max(sfvv, "max_temp", False)
vf_temp_min, vf_temp_max = get_efficiency_stat_min_max(vfvv, "max_temp", False)

sf_ips_min, sf_ips_max = get_efficiency_stat_min_max(sfvv, "ips_conservative", False)
vf_ips_min, vf_ips_max = get_efficiency_stat_min_max(vfvv, "ips_potential", False)

print(f"SF EDP range: {sf_edp_min}% to {sf_edp_max}%")
print(f"SF IPS range: {sf_ips_min}% to {sf_ips_max}%")
print(f"VF EDP range: {vf_edp_min}% to {vf_edp_max}%")
print(f"VF IPS range: {vf_ips_min}% to {vf_ips_max}%")
print(f"SF Max temperature range: {sf_temp_min} to {sf_temp_max} celsius")
print(f"VF Max temperature range: {vf_temp_min} to {vf_temp_max} celsius")

# Represent EDP over baseline in terms of just percentages, and then absolute values
# Same for IPS

sf_df = pd.DataFrame.from_dict(sfvv, orient="index")
vf_df = pd.DataFrame.from_dict(vfvv, orient="index")

sf_df["edp_constant"] = -sf_df["edp_constant"]
vf_df["edp_potential"] = -vf_df["edp_potential"]
sf_df["energy_constant"] = -sf_df["energy_constant"]
vf_df["energy_potential"] = -vf_df["energy_potential"]

sf_df["edp_etc_norm"] = sf_df["edp_etc_value"] / sf_df["edp_base_value"]
sf_df["edp_base_norm"] = sf_df["edp_base_value"] / sf_df["edp_base_value"]

sf_df["ips_etc_norm"] = sf_df["ips_etc_value"] / sf_df["ips_base_value"]
sf_df["ips_base_norm"] = sf_df["ips_base_value"] / sf_df["ips_base_value"]

vf_df["edp_etc_norm"] = vf_df["edp_etc_value"] / vf_df["edp_base_value"]
vf_df["edp_base_norm"] = vf_df["edp_base_value"] / vf_df["edp_base_value"]

vf_df["ips_etc_norm"] = vf_df["ips_etc_value"] / vf_df["ips_base_value"]
vf_df["ips_base_norm"] = vf_df["ips_base_value"] / vf_df["ips_base_value"]


def plot_stats(
    df: pd.DataFrame,
    stats: list,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    value_name: str = "EDP",
    figure_name: str = "",
    threshold: float = None,
):
    df_melt = (
        df[stats]
        .reset_index()
        .melt(
            id_vars="index", value_vars=stats, var_name="metric", value_name=value_name
        )
    )
    df_melt = df_melt.rename(columns={"index": "program_name"})
    df_melt["metric"] = df_melt["metric"].replace(
        {
            "edp_etc_norm_sf": "Stable-frequency EDP",
            "edp_etc_norm_vf": "Variable-frequency EDP",
            "edp_etc_norm": "Our approach normalised EDP",
            "edp_base_norm": "Baseline normalised EDP",
            "ips_etc_value": "Our approach IPS",
            "ips_base_value": "Baseline IPS",
            "ips_etc_norm": "Our approach normalised IPS",
            "ips_base_norm": "Baseline normalised IPS",
            "edp_constant": "Stable-frequency EDP %improvement",
            "edp_potential": "Variable-frequency EDP %improvement",
            "ips_conservative": "Stable-frequency IPS %improvement",
            "ips_potential": "Variable-frequency IPS %improvement",
            "max_temp": "Maximum temperature",
            "avg_temp": "Weighted average temperature",
            "max_freq": "Maximum clock rate",
            "avg_freq": "Average clock rate",
            "energy_impr_sf": "Stable-frequency energy %improvement",
            "energy_impr_vf": "Variable-frequency energy %improvement",
        }
    )

    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_alpha(0)

    sns.barplot(data=df_melt, x="program_name", y=value_name, hue="metric", ax=ax)
    ax.set_title(title, fontsize=28, pad=16)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(ylabel, fontsize=26)
    ax.tick_params(axis="both", labelsize=24)
    ax.set_facecolor("none")
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=4, fontsize=26)

    if threshold is not None:
        ax.axhline(
            y=threshold, color="red", linestyle="--", label="Safeguard temperature"
        )
        ax.set_ylim(0, 90)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if figure_name:
        plt.savefig(figure_name, format="eps", dpi=300, bbox_inches="tight")

    plt.show()


sf_df["energy_impr"] = sf_df["energy_constant"]
vf_df["energy_impr"] = vf_df["energy_potential"]


unified_df = sf_df.merge(
    vf_df, left_index=True, right_index=True, suffixes=("_sf", "_vf")
)
unified_df["edp_base_norm"] = unified_df["edp_base_norm_sf"]

plot_stats(
    unified_df,
    ["edp_etc_norm_sf", "edp_etc_norm_vf", "edp_base_norm"],
    "Stable vs Variable frequency normalised EDP comparison",
    "EDP value",
    "Program",
    "EDP",
    "./results/uf_edp_normalised_compare.eps",
)
plot_stats(
    unified_df,
    ["energy_impr_sf", "energy_impr_vf"],
    "Stable vs Variable frequency energy improvement",
    "Energy percentage improvement",
    "Program",
    "Energy",
    "./results/uf_energy_improve.eps",
)

plot_stats(
    sf_df,
    ["ips_etc_value", "ips_base_value"],
    "Stable frequency IPS value comparison",
    "IPS value",
    "Program",
    "IPS",
    "./results/sf_ips_value_compare.eps",
)
plot_stats(
    vf_df,
    ["ips_etc_value", "ips_base_value"],
    "Variable frequency IPS value comparison",
    "IPS value",
    "Program",
    "IPS",
    "./results/vf_ips_value_compare.eps",
)

plot_stats(
    sf_df,
    ["edp_etc_norm", "edp_base_norm"],
    "Stable frequency normalised EDP comparison",
    "EDP value",
    "Program",
    "EDP",
    "./results/sf_edp_value_norm_compare.eps",
)
plot_stats(
    sf_df,
    ["ips_etc_norm", "ips_base_norm"],
    "Stable frequency normalised IPS comparison",
    "IPS value",
    "Program",
    "IPS",
    "./results/sf_ips_value_norm_compare.eps",
)
plot_stats(
    vf_df,
    ["edp_etc_norm", "edp_base_norm"],
    "Variable frequency normalised EDP comparison",
    "EDP value",
    "Program",
    "EDP",
    "./results/vf_edp_value_norm_compare.eps",
)
plot_stats(
    vf_df,
    ["ips_etc_norm", "ips_base_norm"],
    "Variable frequency normalised IPS comparison",
    "IPS value",
    "Program",
    "IPS",
    "./results/vf_ips_value_norm_compare.eps",
)

plot_stats(
    sf_df,
    ["edp_constant"],
    "Stable frequency EDP improvement",
    "EDP improvement",
    "Program",
    "EDP",
    "./results/sf_edp_percent_improvement.eps",
)
plot_stats(
    sf_df,
    ["ips_conservative"],
    "Stable frequency IPS improvement",
    "IPS improvement",
    "Program",
    "IPS",
    "./results/sf_ips_percent_improvement.eps",
)

plot_stats(
    vf_df,
    ["edp_potential"],
    "Variable frequency EDP comparison",
    "EDP improvement",
    "Program",
    "EDP",
    "./results/vf_edp_percent_improvement.eps",
)
plot_stats(
    vf_df,
    ["ips_potential"],
    "Variable frequency IPS comparison",
    "IPS improvement",
    "Program",
    "IPS",
    "./results/vf_ips_percent_improvement.eps",
)

plot_stats(
    sf_df,
    ["avg_temp", "max_temp"],
    "Stable frequency temperature",
    "Temperature (Celsius)",
    "Program",
    "temperature",
    "./results/sf_temp.eps",
    85,
)
plot_stats(
    vf_df,
    ["avg_temp", "max_temp"],
    "Variable frequency temperature",
    "Temperature (Celsius)",
    "Program",
    "temperature",
    "./results/vf_temp.eps",
    85,
)

plot_stats(
    vf_df,
    ["avg_freq", "max_freq"],
    "Variable frequency clock rate",
    "Frequency (GHz)",
    "Program",
    "frequency",
    "./results/vf_clock_rate.eps",
)
