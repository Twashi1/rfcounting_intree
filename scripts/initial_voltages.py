import argparse
import utils
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial voltages at some voltage level"
    )
    parser.add_argument(
        "--voltage_level",
        type=int,
        default="0",
        help="The voltage level to initialise all block ids to, expecting the index",
    )
    parser.add_argument(
        "--module_index", type=int, default=2, help="The module to base BlockIDs off"
    )
    parser.add_argument(
        "--additional_block",
        type=str,
        default="PerBlockAdditional.csv",
        help="Additional block data where we will get the block ids from",
    )
    parser.add_argument(
        "--output", type=str, default="VoltageLevels.csv", help="Filename of output"
    )
    args = parser.parse_args()

    # TODO: validation on voltage_level
    block_ids = utils.get_block_ids(args.module_index, args.additional_block)
    utils.init_voltages(args.output, args.voltage_level, block_ids)


if __name__ == "__main__":
    main()
