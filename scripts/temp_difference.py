import argparse
import utils
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Given heat data of ETC approach and baseline, compare temperature"
    )
    parser.add_argument(
        "--etc_heat", type=str, help="Program heat data from our approach"
    )
    parser.add_argument(
        "--baseline_heat", type=str, help="Program heat data of the baseline"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./scripts/configs.cfg",
        help="Config file to get the voltages and target clock frequency from",
    )
    parser.add_argument(
        "--module_index",
        type=int,
        default=2,
        help="The module to get for block additional data",
    )
    parser.add_argument(
        "--out_prefix", type=str, help="Name of the program/prefix to output under"
    )
    args = parser.parse_args()

    # Load per-basic block heat data
    etc_heat = utils.load_program_heats(args.etc_heat)
    baseline_heat = utils.load_program_heats(args.baseline_heat)

    df = etc_heat.merge(
        baseline_heat, on="block_id", how="outer", suffixes=("_etc", "_baseline")
    )

    df["temp_diff"] = df["temp_max_etc"] - df["temp_max_baseline"]
    # Converting to celsius
    df["temp_max_etc"] = df["temp_max_etc"] - 273.15
    df["temp_max_baseline"] = df["temp_max_baseline"] - 273.15
    output = df[["block_id", "temp_max_etc", "temp_max_baseline", "temp_diff"]]

    output.to_csv(f"{args.out_prefix}_ProgramTemperatureChange.csv", index=False)


if __name__ == "__main__":
    main()
