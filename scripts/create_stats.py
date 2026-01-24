import argparse
import utils

def main():
    parser = argparse.ArgumentParser(description="Generate standard stat files from some input")
    parser.add_argument("--input_file", help="Some input .csv file or gem5 output stats.txt")
    parser.add_argument("--output", type=str, default="out_standard", help="The name to give the output file")
    parser.add_argument("--module_index", type=int, default=2, help="The module index to include (expecting 2 usually for the profiled run)")
    parser.add_argument("--path_index", type=int, default=-1, help="The block or path to include if applicable, or default to include all")
    parser.add_argument("--take_sum", type=int, default=0, help="Should we take a sum over blocks/paths, or should we keep them separate")
    args = parser.parse_args()

    utils.create_standard_stat_file(args.input_file, args.output, args.module_index, args.path_index, args.take_sum)

if __name__ == "__main__":
    main()
