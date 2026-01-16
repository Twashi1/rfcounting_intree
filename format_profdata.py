import argparse

def format_text(profdata_in, output):
    with open(profdata_in, "r") as f:
        lines = f.readlines()

    with open(output, "w+") as f:
        current_function_name = ""
        current_file_name = ""

        f.write("file,function_name,block_number,count\n")

        for line in lines:
            if line.startswith("Counters:"):
                continue

            # probably the name of function?
            if ".c:" in line:
                file_name, function_name, _rest = line.split(":")
                print(f"Got line: {file_name=} {function_name=} {_rest=}")

                current_file_name = file_name.strip()
                current_function_name = function_name.strip()

            if line.lstrip().startswith("Block counts: "):
                _, block_counts = line.split(":")
                block_counts = block_counts.strip()
                # TODO: real bad
                some_array = eval(block_counts)

                print(f"Got counts: {some_array} of {type(some_array)}")

                if current_file_name:
                    for i, count in enumerate(some_array):
                        f.write(f"{current_file_name},{current_function_name},{i},{count}\n")

                current_function_name = ""
                current_file_name = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format the output of the profdata command")
    parser.add_argument("profdata_out", help="Should be profdata.txt", default="profdata.txt")
    parser.add_argument("output", help="Output file (csv)", default="outprof.csv")
    args = parser.parse_args()

    format_text(args.profdata_out, args.output)
