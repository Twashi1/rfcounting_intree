import utils
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Create xml file based on Alpha21364 given some data.")
    parser.add_argument("--stats", help="Path to some _STD.csv file")
    parser.add_argument("--input_xml", type=str, default="mcpat_inputs/Alpha21364.xml", help="Path to input XML (expecting Alpha21364.xml) file")
    parser.add_argument("--output_xml", help="Name of output XML file")
    parser.add_argument("--input_cfg", type=str, default="mcpat_inputs/mcpat.cfg", help="McPAT config file")
    args = parser.parse_args()

    loaded_stats = utils.load_standard_stat_file(args.stats)
    loaded_cfg = utils.load_cfg(args.input_cfg)
    utils.modify_xml(args.input_xml, args.output_xml, loaded_stats, loaded_cfg)

if __name__ == "__main__":
    main()
