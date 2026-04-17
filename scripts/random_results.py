# Generating random results as needed
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sfvv = utils.load_efficiency_stats("./results/sfvv_per_program.txt")
vfvv = utils.load_efficiency_stats("./results/vfvv_per_program.txt")

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

sf_edp_min, sf_edp_max = get_efficiency_stat_min_max(sfvv, "edp_constant", True) 
vf_edp_min, vf_edp_max = get_efficiency_stat_min_max(vfvv, "edp_potential", True)
sf_temp_min, sf_temp_max = get_efficiency_stat_min_max(sfvv, "max_temp", False)

sf_ips_min, sf_ips_max = get_efficiency_stat_min_max(sfvv, "ips_conservative", False) 
vf_ips_min, vf_ips_max = get_efficiency_stat_min_max(vfvv, "ips_potential", False) 

print(f"SF EDP range: {sf_edp_min}% to {sf_edp_max}%")
print(f"SF IPS range: {sf_ips_min}% to {sf_ips_max}%")
print(f"VF EDP range: {vf_edp_min}% to {vf_edp_max}%")
print(f"VF IPS range: {vf_ips_min}% to {vf_ips_max}%")
print(f"SF Max temperature range: {sf_temp_min} to {sf_temp_max} celsius")

# Represent EDP over baseline in terms of just percentages, and then absolute values
# Same for IPS

sf_df = pd.DataFrame.from_dict(sfvv, orient="index")
vf_df = pd.DataFrame.from_dict(vfvv, orient="index")

sf_df["edp_constant"] = -sf_df["edp_constant"]
vf_df["edp_potential"] = -vf_df["edp_potential"]

sf_df["edp_etc_norm"] = sf_df["edp_etc_value"] / sf_df["edp_base_value"]
sf_df["edp_base_norm"] = sf_df["edp_base_value"] / sf_df["edp_base_value"]

# vf_df["edp_etc_norm"] = vf_df["edp_etc_value"] / vf_df["edp_base_value"]
# vf_df["edp_base_norm"] = vf_df["edp_base_value"] / vf_df["edp_base_value"]

def plot_stats(df: pd.DataFrame, stats: list,
                     title: str = "", xlabel: str = "", ylabel: str = ""):
    ax = df[stats].plot(kind='bar', legend=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(stats)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_stats(sf_df, ["edp_etc_norm", "edp_base_norm"], "Stable frequency normalised EDP value comparison", "EDP value", "Program")
# plot_stats(vf_df, ["edp_etc_norm", "edp_base_norm"], "Variable frequency normalised EDP value comparison", "EDP value", "Program")

plot_stats(sf_df, ["edp_constant"], "Stable frequency EDP improvement", "EDP improvement", "Program")
plot_stats(sf_df, ["ips_conservative"], "Stable frequency IPS improvement", "IPS improvement", "Program")
plot_stats(sf_df, ["avg_temp", "max_temp"], "Stable frequency temperature", "Temperature (Celsius)", "Program")

plot_stats(vf_df, ["edp_potential"], "Variable frequency EDP comparison", "EDP improvement", "Program")
plot_stats(vf_df, ["ips_potential"], "Variable frequency IPS comparison", "IPS improvement", "Program")
plot_stats(vf_df, ["avg_temp", "max_temp"], "Variable frequency temperature", "Temperature (Celsius)", "Program")
plot_stats(vf_df, ["avg_freq", "max_freq"], "Variable frequency clock rate", "Frequency (GHz)", "Program")

plot_stats(sf_df, ["edp_etc_value", "edp_base_value"], "Stable frequency EDP value comparison", "EDP value", "Program")
plot_stats(sf_df, ["ips_etc_value", "ips_base_value"], "Stable frequency IPS value comparison", "IPS value", "Program")

#plot_stats(vf_df, ["edp_etc_value", "edp_base_value"], "Variable frequency EDP value comparison", "EDP value", "Program")
#plot_stats(vf_df, ["ips_etc_value", "ips_base_value"], "Variable frequency IPS value comparison", "IPS value", "Program")
