import utils
import pandas as pd
import argparse
import seaborn as sns
import os
from matplotlib import pyplot as plt


def create_unified_stats_per_block(
    program_name: str,
    heat_voltage_stats: pd.DataFrame,
    temp_diff_stats: pd.DataFrame,
) -> pd.DataFrame:
    df = heat_voltage_stats.merge(temp_diff_stats, on="block_id", how="outer")
    df = df[
        [
            "block_id",
            "temp_max",
            "execution_time",
            "dvs_calling_count",
            "required_voltage_value",
            "obtained_frequency",
        ]
    ].copy()

    df = df.rename(
        {
            "required_voltage_value": "voltage",
            "obtained_frequency": "max_frequency_allowable",
        }
    )

    df.to_csv(f"./output_stats/{program_name}_UnifiedStats.csv")

    return df


def create_unified_program_stats(efficiency_stats: dict) -> pd.DataFrame:
    df_dict = {
        "program_name": [],
        "edp_percent": [],
        "energy_percent": [],
        "ips_percent": [],
    }

    for program_name, program_data in efficiency_stats.items():
        df_dict["program_name"].append(program_name)
        df_dict["edp_percent"].append(-program_data["edp_potential"])
        df_dict["energy_percent"].append(-program_data["energy_potential"])
        df_dict["ips_percent"].append(program_data["ips_potential"])

    df = pd.DataFrame(df_dict)

    df.to_csv(f"./output_stats/overall.csv", index=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Given per-program efficiency stats, plot difference between baseline and our approach"
    )
    parser.add_argument(
        "--efficiency_stats",
        type=str,
        default="efficiencyStats.txt",
        help="Efficiency stats",
    )
    parser.add_argument(
        "--block_heats",
        type=str,
        default="block_heats",
        help="Folder with all block heats for us to grab the temperature differences from",
    )
    args = parser.parse_args()

    # TODO: take as argument
    os.makedirs("./output_stats/", exist_ok=True)

    stats_dict = utils.load_efficiency_stats(args.efficiency_stats)

    unified_program = create_unified_program_stats(stats_dict)
    utils.info(f"{unified_program}")
    unified_block_stats = {}

    # TODO: load block-heat
    # ProgramVoltages, and ProgramTemperatureChange

    for program_name in stats_dict:
        path = f"./{args.block_heats}/{program_name}"
        program_voltages = f"{path}_ProgramHeatVoltages.csv"
        program_temp_diff = f"{path}_ProgramTemperatureChange.csv"

        heat_voltages_df = utils.load_program_heats_voltages(program_voltages)
        temp_diff_df = utils.load_temperature_diff_stats(program_temp_diff)

        unified_per_block = create_unified_stats_per_block(
            program_name, heat_voltages_df, temp_diff_df
        )

        unified_block_stats[program_name] = unified_per_block

    # TODO: require absolute values of EDP/Energy/etc. to plot
    # TODO: output total number of DVS calling points

    # Create barplot of unified program stats
    # TODO: flip edp_percent
    # TODO: add titles to both
    plt.figure()
    sns.barplot(data=unified_program, x="program_name", y="edp_percent")
    plt.title("EDP percentage improvement")

    plt.figure()
    sns.barplot(data=unified_program, x="program_name", y="ips_percent")
    plt.title("IPS percentage differnce")

    plt.show()


if __name__ == "__main__":
    main()
