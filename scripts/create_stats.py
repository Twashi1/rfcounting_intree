import argparse
import utils

def main():
    parser = argparse.ArgumentParser(description="Generate standard stat files from some input")
    parser.add_argument("--input_file", help="Some input .csv file or gem5 output stats.txt")
    parser.add_argument("--output", type=str, default="out_standard", help="The name to give the output file")
    parser.add_argument("--module_index", type=int, default=0, help="The module to analyse if applicable")
    parser.add_argument("--path_index", type=int, default=-1, help="The block or path to analyse if applicable, or leave as default to take a sum")
    args = parser.parse_args()

    utils.create_standard_stat_file(args.input_file, args.output, args.module_index, args.path_index)

if __name__ == "__main__":
    main()
