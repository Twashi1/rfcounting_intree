import utils
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Read required voltages from _ProgramHeatVoltages.csv, and output them to a desired csv"
    )
    parser.add_argument(
        "--tei_voltages",
        help="Path to some _ProgramHeatVoltages.csv file, to read the required voltages/voltage level",
    )
    parser.add_argument(
        "--input_cfg",
        type=str,
        default="scripts/configs.cfg",
        help="General config file",
    )
    parser.add_argument(
        "--out_voltages",
        type=str,
        default="VoltageLevels.csv",
        help="The file to output voltages to",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tei_voltages)
    df["required_voltage_value"] = df["required_voltage_value"].astype(float)

    voltage_df = df[["block_id", "required_voltage"]].copy()
    voltage_df["voltage_level"] = df["required_voltage"]

    # TODO: consider additional transformation where we select the closest voltage that actually has a power trace

    # TODO: test
    voltage_df.to_csv(args.out_voltages, index=False)


if __name__ == "__main__":
    main()
