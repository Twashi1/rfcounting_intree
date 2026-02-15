import utils
import argparse
import pandas as pd
import os

# TODO: horrendous amount of code duplication
# TODO: also deprecated with voltage level changes


def main():
    parser = argparse.ArgumentParser(
        description="Create xml file based on Alpha21364 given some data."
    )
    parser.add_argument("--stats", help="Path to some _STD.csv file")
    parser.add_argument(
        "--input_xml",
        type=str,
        default="mcpat_inputs/Alpha21364.xml",
        help="Path to input XML (expecting Alpha21364.xml) file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mcpat_inputs",
        help="Additional folder to place files within",
    )
    parser.add_argument("--output_xml", help="Prefix of output XML files")
    parser.add_argument(
        "--input_cfg", type=str, default="scripts/configs.cfg", help="McPAT config file"
    )
    args = parser.parse_args()

    # We assume a single _STD.csv file
    # trivially load-able with some utils function
    # note this should be a dataframe with multiple entries, not a series
    # each entry assumed to be a dvs calling point, so we can get power stats per calling point, in all 3 voltage regions
    loaded_stats = utils.load_standard_stat_file(args.stats)
    loaded_cfg = utils.load_cfg(args.input_cfg)
    voltage_levels = utils.load_voltage_levels_from_cfg(loaded_cfg)
    voltage_index_to_generate = utils.get_distributed_voltage_levels(
        voltage_levels, int(loaded_cfg["mcpat"]["COMPUTE_LEVELS"])
    )

    # Generally not a good pattern, but whatever
    for _, row in loaded_stats.iterrows():
        row_data = row.to_dict()

        # TODO: use Pathlib or something
        path = args.output_dir.rstrip("/") + "/"

        os.makedirs(path, exist_ok=True)

        if "block_id" not in row_data:
            raise KeyError(
                "Expected block_id in mcpat row data, given DVS calling point data instead of per machine block?"
            )

        i = row_data["block_id"]

        path += args.output_xml
        path += f"_idx{i:04d}"

        # Really ugly, we're using a counter to distinguish, so its in order of appearance of _STD.csv file
        #   we have no metadata inside the McPAT output to support us, so we have to match these up later
        for voltage_id in voltage_index_to_generate:
            utils.modify_xml(
                args.input_xml,
                path + f"_v{voltage_id}.xml",
                row_data,
                loaded_cfg,
                voltage_id,
            )


if __name__ == "__main__":
    main()
