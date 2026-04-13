import argparse
import utils
import re
import pandas as pd
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Take Gem5 output from alternative approach, and find EDP")
    parser.add_argument("--input_file", help="Gem5 output stats.txt")
    parser.add_argument("--input_cfg", type=str, default="./scripts/configs.cfg", help="The config file")
    parser.add_argument("--output_folder", help="The output folder for McPAT")
    parser.add_argument("--program_name", help="The name of the program being profiled")
    args = parser.parse_args()

    stats_blocks = utils.get_stats_df_gem5_run(args.input_file)
    loaded_cfg = utils.load_cfg(args.input_cfg)

    index = 0

    stat_list = stats_blocks.to_dict(orient="records")

    power_dictionaries = []
    
    os.makedirs(f"./mcpat_inputs/{args.program_name}/", exist_ok=True)

    for i, row in enumerate(stat_list):
        row_dict = row

        voltage = row_dict["average_voltage"]
        frequency = row_dict["average_frequency"]
        print(f"Frequency was: {frequency}")

        input_xml = f"./mcpat_inputs/{args.program_name}/Gem5_{index}.xml"
        output_power_trace = f"./{args.output_folder}/Gem5_{index}.txt"

        print(row_dict)

        utils.create_mcpat_input_xml("./mcpat_inputs/Alpha21364.xml", input_xml, row_dict, loaded_cfg, voltage, frequency / 1_000_000.0)

        # subprocess.run(
        #     ["./run_mcpat_specific.sh", f"Gem5_{index}", f"{args.program_name}"],
        #     check=True,
        # )

        mcpat_dict = utils.mcpat_to_dict(output_power_trace)
        power_dictionaries.append(mcpat_dict)

        index += 1

        print(f"Ran McPAT on {index} out of {len(stats_blocks)}")

    energies = []
    runtimes = []

    for i, power_dict in enumerate(power_dictionaries):
        core_power = utils.get_static_dynamic_power(power_dict, ["Core"])
        stats = stat_list[i]
        cycles = stats[utils.CYCLE_COUNT]
        frequency = stats["average_frequency"]

        # TODO: if no DVFS events occur during an interval, we don't record the average frequency/voltage
        if frequency == 0.0:
            frequency = last_good_frequency
            utils.warn(f"Frequency was 0.0 for block {i}")
            continue

        last_good_frequency = frequency

        runtime = cycles / frequency

        energies.append(runtime * core_power)
        runtimes.append(runtime)

    overall_edp = sum(energies) * sum(runtimes)
    print(f"EDP of approach: {overall_edp}")

    return mcpat_dict

if __name__ == "__main__":
    main()
