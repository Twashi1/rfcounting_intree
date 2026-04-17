import argparse
import utils
import re
import pandas as pd
import subprocess
import os
import configparser

def get_stats(input_file: str, loaded_cfg: configparser.ConfigParser, output_folder: str, program_name: str, dvfs: bool):
    stats_blocks = utils.get_stats_df_gem5_run(input_file)

    if dvfs:
        program_name = program_name + "_dvfs"

    index = 0

    stat_list = stats_blocks.to_dict(orient="records")

    power_dictionaries = []
    
    os.makedirs(f"./mcpat_inputs/{program_name}/", exist_ok=True)

    for i, row in enumerate(stat_list):
        row_dict = row

        voltage = row_dict["average_voltage"]
        frequency = row_dict["average_frequency"]

        input_xml = f"./mcpat_inputs/{program_name}/Gem5_{index}.xml"
        output_power_trace = f"./{output_folder}/Gem5_{index}.txt"

        utils.create_mcpat_input_xml("./mcpat_inputs/Alpha21364.xml", input_xml, row_dict, loaded_cfg, voltage, frequency / 1_000_000.0)

        # TODO: check if file already exists, don't recreate
        subprocess.run(
            ["./run_mcpat_specific.sh", f"Gem5_{index}", f"{program_name}"],
            check=True,
        )

        mcpat_dict = utils.mcpat_to_dict(output_power_trace)
        power_dictionaries.append(mcpat_dict)

        index += 1

        power = utils.get_static_dynamic_power(mcpat_dict, ["Core"])
        print(f"Ran McPAT on {index} out of {len(stats_blocks)}; got power: {power}")


    return power_dictionaries, stat_list

def get_edp(power_dictionaries: list, stat_list: list):
    energies = []
    runtimes = []

    for i, power_dict in enumerate(power_dictionaries):
        core_power = utils.get_static_dynamic_power(power_dict, ["Core"])
        stats = stat_list[i]
        cycles = stats[utils.CYCLE_COUNT]
        frequency = stats["average_frequency"]

        # TODO: if no DVFS events occur during an interval, we don't record the average frequency/voltage
        if frequency == 0.0:
            utils.warn(f"Frequency was 0.0 for block {i}")
            continue

        runtime = cycles / frequency

        energies.append(runtime * core_power)
        runtimes.append(runtime)

    overall_edp = sum(energies) * sum(runtimes)

    return overall_edp

def get_ips(stat_list: list):
    instructions = []
    runtimes = []

    for i in range(len(stat_list)):
        stats = stat_list[i]
        cycles = stats[utils.CYCLE_COUNT]
        frequency = stats["average_frequency"]

        # TODO: if no DVFS events occur during an interval, we don't record the average frequency/voltage
        if frequency == 0.0:
            utils.warn(f"Frequency was 0.0 for block {i}")
            continue

        runtime = cycles / frequency

        instructions.append(stats[utils.INSTR_COUNT])
        runtimes.append(runtime)

    overall_ips = sum(instructions) / sum(runtimes)

    return overall_ips 

# TODO: put in utils
def percent_increase(new: float, old: float) -> float:
    return (new - old) / old * 100.0

def main():
    parser = argparse.ArgumentParser(description="Take Gem5 output from alternative approach, and find EDP")
    parser.add_argument("--input_file_dvfs", help="Gem5 DVFS output stats.txt")
    parser.add_argument("--input_file_base", help="Gem5 non-DVFS output stats.txt")
    parser.add_argument("--input_cfg", type=str, default="./scripts/configs.cfg", help="The config file")
    parser.add_argument("--output_folder", help="The output folder for McPAT")
    parser.add_argument("--program_name", help="The name of the program being profiled")
    args = parser.parse_args()

    loaded_cfg = utils.load_cfg(args.input_cfg)

    power_dict_dvfs, stat_list_dvfs = get_stats(args.input_file_dvfs, loaded_cfg, args.output_folder, args.program_name, True)
    power_dict_base, stat_list_base = get_stats(args.input_file_base, loaded_cfg, args.output_folder, args.program_name, False)

    edp_dvfs = get_edp(power_dict_dvfs, stat_list_dvfs)
    edp_base = get_edp(power_dict_base, stat_list_base)

    ips_dvfs = get_ips(stat_list_dvfs)
    ips_base = get_ips(stat_list_base)

    edp_change = percent_increase(edp_dvfs, edp_base)
    ips_change = percent_increase(ips_dvfs, ips_base)

    print(f"{edp_dvfs=} {edp_base=} {ips_dvfs=} {ips_base=}")
    output_text = f"[{args.program_name}] EDP change: {edp_change:.4f}% (lower better), IPS change: {ips_change:.4f}% (higher better)"

    print(output_text)

    with open("gem5_output_efficiency.txt", "a") as f:
        f.write(output_text + "\n")

if __name__ == "__main__":
    main()
