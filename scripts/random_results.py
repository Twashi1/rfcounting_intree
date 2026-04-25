# Generating random results as needed
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Can set theme here
sns.set_theme(style="whitegrid")

sfvv = utils.load_efficiency_stats("./results/sfvv_per_program.txt")
vfvv = utils.load_efficiency_stats("./results/vfvv_per_program.txt")

def load_program_data(program_names, directory="block_heats"):
    results = {}

    for program_name in program_names:
        df = utils.load_program_heats_voltages(f"./{directory}/{program_name}_ProgramHeatVoltages.csv")

        total_dvfs = df["dvs_calling_count"].sum()
        total_execution_time = df["execution_time"].sum()

        results[program_name] = {
                "dvfs_calling_count": total_dvfs,
                "execution_time": total_execution_time
        }

    return results

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

print(sfvv_additional)

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

sf_df["edp_etc_norm"] = sf_df["edp_etc_value"] / sf_df["edp_base_value"]
sf_df["edp_base_norm"] = sf_df["edp_base_value"] / sf_df["edp_base_value"]

sf_df["ips_etc_norm"] = sf_df["ips_etc_value"] / sf_df["ips_base_value"]
sf_df["ips_base_norm"] = sf_df["ips_base_value"] / sf_df["ips_base_value"]

vf_df["edp_etc_norm"] = vf_df["edp_etc_value"] / vf_df["edp_base_value"]
vf_df["edp_base_norm"] = vf_df["edp_base_value"] / vf_df["edp_base_value"]

vf_df["ips_etc_norm"] = vf_df["ips_etc_value"] / vf_df["ips_base_value"]
vf_df["ips_base_norm"] = vf_df["ips_base_value"] / vf_df["ips_base_value"]

def plot_stats(df: pd.DataFrame, stats: list,
               title: str = "", ylabel: str = "", xlabel: str = "", value_name: str = "EDP", figure_name: str = "", threshold: float = None):
    df_melt = df[stats].reset_index().melt(id_vars="index", value_vars=stats, var_name="metric", value_name=value_name)
    df_melt = df_melt.rename(columns={"index": "program_name"})
    df_melt["metric"] = df_melt["metric"].replace({
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
        "avg_freq": "Average clock rate"
    })

    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_alpha(0)

    sns.barplot(data=df_melt, x="program_name", y=value_name, hue="metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("none")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="center left")

    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle="--", label="Safeguard temperature")
        ax.set_ylim(0, 90)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if figure_name:
        plt.savefig(figure_name, dpi=300, bbox_inches="tight")

    plt.show()

unified_df = sf_df.reset_index().merge(vf_df.reset_index(), on="index", suffixes=("_sf", "_vf"))
unified_df["edp_base_norm"] = unified_df["edp_base_norm_sf"]

plot_stats(unified_df, ["edp_etc_norm_sf", "edp_etc_norm_vf", "edp_base_norm"], "Stable vs Variable frequency normalised EDP comparison", "EDP value", "Program", "EDP", "./results/uf_edp_normalised_compare.png")

plot_stats(sf_df, ["ips_etc_value", "ips_base_value"], "Stable frequency IPS value comparison", "IPS value", "Program", "IPS", "./results/sf_ips_value_compare.png")
plot_stats(vf_df, ["ips_etc_value", "ips_base_value"], "Variable frequency IPS value comparison", "IPS value", "Program", "IPS", "./results/vf_ips_value_compare.png")

plot_stats(sf_df, ["edp_etc_norm", "edp_base_norm"], "Stable frequency normalised EDP comparison", "EDP value", "Program", "EDP", "./results/sf_edp_value_norm_compare.png")
plot_stats(sf_df, ["ips_etc_norm", "ips_base_norm"], "Stable frequency normalised IPS comparison", "IPS value", "Program", "IPS", "./results/sf_ips_value_norm_compare.png")
plot_stats(vf_df, ["edp_etc_norm", "edp_base_norm"], "Variable frequency normalised EDP comparison", "EDP value", "Program", "EDP", "./results/vf_edp_value_norm_compare.png")
plot_stats(vf_df, ["ips_etc_norm", "ips_base_norm"], "Variable frequency normalised IPS comparison", "IPS value", "Program", "IPS", "./results/vf_ips_value_norm_compare.png")

plot_stats(sf_df, ["edp_constant"], "Stable frequency EDP improvement", "EDP improvement", "Program", "EDP", "./results/sf_edp_percent_improvement.png")
plot_stats(sf_df, ["ips_conservative"], "Stable frequency IPS improvement", "IPS improvement", "Program", "IPS", "./results/sf_ips_percent_improvement.png")

plot_stats(vf_df, ["edp_potential"], "Variable frequency EDP comparison", "EDP improvement", "Program", "EDP", "./results/vf_edp_percent_improvement.png")
plot_stats(vf_df, ["ips_potential"], "Variable frequency IPS comparison", "IPS improvement", "Program", "IPS", "./results/vf_ips_percent_improvement.png")

plot_stats(sf_df, ["avg_temp", "max_temp"], "Stable frequency temperature", "Temperature (Celsius)", "Program", "temperature", "./results/sf_temp.png", 85)
plot_stats(vf_df, ["avg_temp", "max_temp"], "Variable frequency temperature", "Temperature (Celsius)", "Program", "temperature", "./results/vf_temp.png", 85)

plot_stats(vf_df, ["avg_freq", "max_freq"], "Variable frequency clock rate", "Frequency (GHz)", "Program", "frequency", "./results/vf_clock_rate.png")

# plot_stats(sf_df, ["edp_etc_value", "edp_base_value"], "Stable frequency EDP value comparison", "EDP value", "Program")
