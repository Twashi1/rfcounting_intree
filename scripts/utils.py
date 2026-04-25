import csv
import xml.etree.ElementTree as ET
import re
import copy
import math
import subprocess
import inspect
import os
from numpy import int_
import pandas as pd
import configparser
from collections import defaultdict
from pathlib import Path

MCPAT_CFG_MODULE_NAME = "mcpat"
MCPAT_CLOCK_RATE_MHZ = "CLOCK_RATE"
MCPAT_TEMP = "TEMP"
MCPAT_NODE_SIZE = "NODE_SIZE"
MCPAT_BASELINE_VOLTAGE = "BASELINE_VOLTAGE_LEVEL"
MCPAT_VOLTAGE_UPWARDS_ADJUSTMENT = "VOLTAGE_UPWARDS_ADJUSTMENT"
MCPAT_TEMP_LOW = "TEMP_LOW"
MCPAT_TEMP_OPTIM = "TEMP_OPTIM"
MCPAT_TEMP_SAFEGUARD_MAX = "TEMP_SAFEGUARD_MAX"
MCPAT_ATTEMPT_TEMP_OPTIM = "ATTEMPT_TEMPERATURE_OPTIM"
MCPAT_ROUND_UP = "ROUND_VOLTAGE_UP"
MCPAT_FREQUENCY_PRECISION = "FREQUENCY_PRECISION"
MCPAT_FREQUENCY_LIMIT = "FREQUENCY_LIMIT"

HOTSPOT_MODULE_NAME = "hotspot"
HOTSPOT_INCLUDE_RESIDUALS = "INCLUDE_RESIDUALS"
HOTSPOT_HEATSINK_OFFSET = "HEATSINK_OFFSET"

# TODO: implement rest of these functions and use them
# def config_clock_rate_mhz(config: configparser.ConfigParser) -> float:
#     return float(config[MCPAT_CFG_MODULE_NAME][MCPAT_CLOCK_RATE_MHZ])
#
#
# def config_temp(config: configparser.ConfigParser) -> float:
#     return float(config[MCPAT_CFG_MODULE_NAME][MCPAT_TEMP])


# NOTE: should be an enum but it'd be difficult to use as keys
MODULE_NAME = "module_name"
FUNCTION_NAME = "function_name"
BLOCK_NAME = "block_name"

PATH_INDEX = "path_index"
IS_ENTRY = "is_entry"
IS_EXIT = "is_exit"

L1D_HITS = "l1d_hits"
L1D_MISS = "l1d_miss"
L1I_HITS = "l1i_hits"
L1I_MISS = "l1i_miss"
L2_HITS = "l2_hits"
L2_MISS = "l2_miss"

CYCLE_COUNT = "cycle_count"
INSTR_COUNT = "instr_count"
INT_INSTR_COUNT = "int_instr_count"
FLOAT_INSTR_COUNT = "float_instr_count"
BRANCH_INSTR_COUNT = "branch_instr_count"
LOADS = "loads"
STORES = "stores"
FREQ = "freq"
INT_REGFILE_READS = "int_regfile_reads"
INT_REGFILE_WRITES = "int_regfile_writes"
FLOAT_REGFILE_READS = "float_regfile_reads"
FLOAT_REGFILE_WRITES = "float_regfile_writes"
FUNCTION_CALLS = "function_calls"
CONTEXT_SWITCHES = "context_switches"
MUL_ACCESS = "mul_access"
FP_ACCESS = "fp_access"
IALU_ACCESS = "ialu_access"

IDLE_CYCLES = "idle_cycles"
BUSY_CYCLES = "busy_cycles"
COMMITTED_INSTR = "committed_instr"
COMMITTED_INT_INSTR = "committed_int_instr"
COMMITTED_FLOAT_INSTR = "committed_float_instr"
BRANCH_MISPREDICTIONS = "branch_mispredictions"
ROB_READS = "rob_reads"
ROB_WRITES = "rob_writes"
RENAME_READS = "rename_reads"
RENAME_WRITES = "rename_writes"
FP_RENAME_READS = "fp_rename_reads"
FP_RENAME_WRITES = "fp_rename_writes"
INST_WINDOW_WRITES = "inst_window_writes"
INST_WINDOW_READS = "inst_window_reads"
INST_WINDOW_WAKEUP_ACCESSES = "inst_window_wakeup_accesses"
FP_INST_WINDOW_READS = "fp_inst_window_reads"
FP_INST_WINDOW_WRITES = "fp_inst_window_writes"
FP_INST_WINDOW_WAKEUP_ACCESSES = "fp_inst_window_wakeup_accesses"
CDB_MUL_ACCESSES = "cdb_mul_accesses"
CDB_ALU_ACCESSES = "cdb_alu_accesses"
CDB_FP_ACCESSES = "cdb_fp_accesses"
BTB_READS = "btb_reads"
BTB_WRITES = "btb_writes"
VOLTAGE_TIME_PRODUCT = "voltage_time_product"
DVFS_CALL_COUNT = "dvfs_call_count"
AVERAGE_VOLTAGE = "average_voltage"
AVERAGE_FREQUENCY = "average_frequency"
IPC = "ipc"

MCPAT_DESIRED_UNITS = [
    "Core",
    "L2",
    "Instruction Fetch Unit",
    "Instruction Cache",
    "Branch Target Buffer",
    "Branch Predictor",
    "Instruction Buffer",
    "Instruction Decoder",
    "Renaming Unit",
    "Int Front End RAT",
    "FP Front End RAT",
    "Int Retire RAT",
    "FP Retire RAT",
    "FP Free List",
    "Free List",  # assuming this is only integer free list?
    "Load Store Unit",
    "Data Cache",
    "LoadQ",
    "StoreQ",
    "Memory Management Unit",
    "Itlb",
    "Dtlb",
    "Execution Unit",
    "Register Files",
    "Integer RF",
    "Floating Point RF",
    "Instruction Scheduler",
    "Instruction Window",  # assuming this is only integer instruction window?
    "FP Instruction Window",
    "Integer ALU",
    "Floating Point Unit",
    "Results Broadcast Bus",
]


def _grab_metadata():
    current_frame = inspect.currentframe()

    assert current_frame is not None
    assert current_frame.f_back is not None
    assert current_frame.f_back.f_back is not None

    frame = current_frame.f_back.f_back
    line = frame.f_lineno
    script = os.path.basename(frame.f_globals["__file__"])

    return script, line


def info(message):
    script, line = _grab_metadata()
    print(f"[INFO {script}:{line}] {message}")


def warn(message):
    script, line = _grab_metadata()
    print(f"[WARN {script}:{line}] {message}")


def error(message):
    script, line = _grab_metadata()
    print(f"[ERROR {script}:{line}] {message}")


def fatal(message):
    script, line = _grab_metadata()
    print(f"[FATAL {script}:{line}] {message}")
    raise RuntimeError(f"[FATAL {script}:{line}] {message}")


def load_efficiency_stats(efficiency_stats: str) -> dict:
    """
    Return dictionary of program name, stats
    "stats" in turn is a dictionary of stat name, stat value
    """
    results = {}
    program_results = {}

    program_name_pattern = re.compile(r"^\s*Test name:\s+(.+)$")
    stat_value_pattern = re.compile(r"^(.*?):\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)%?$")

    with open(efficiency_stats, "r") as f:
        current_program = None

        for line in f.readlines():
            line = line.rstrip()

            print(line)

            if line.startswith("#"):
                continue

            m = program_name_pattern.search(line)
            if m:
                value = m.group(1)

                info(f"Reading efficiency stats for program {value}")

                if current_program is not None:
                    results[current_program] = program_results
                    program_results = {}

                current_program = value
                continue

            m = stat_value_pattern.search(line)
            if m:
                label = m.group(1)
                value = float(m.group(2))

                program_results[label] = value

        if current_program is not None:
            results[current_program] = program_results
            program_results = {}

    # Brief rename for more friendly names
    renames = {
        "EDP@Constant": "edp_constant",
        "EDP@Potential": "edp_potential",
        "EDP@ETCValue": "edp_etc_value",
        "EDP@BaseValue": "edp_base_value",
        "Energy@Constant": "energy_constant",
        "Energy@Potential": "energy_potential",
        "IPS@Conservative": "ips_conservative",
        "IPS@Potential": "ips_potential",
        "IPS@ETCValue": "ips_etc_value",
        "IPS@BaseValue": "ips_base_value",
        "MaxFreq": "max_freq",
        "AverageFreq": "avg_freq",
        "MaxTemp": "max_temp",
        "AverageTemp": "avg_temp",
    }

    renamed_results = {}

    for program_name, program_data in results.items():
        renamed_data = {}

        for key, value in program_data.items():
            new_key = renames[key]
            renamed_data[new_key] = value

        renamed_results[program_name] = renamed_data

    return renamed_results


def load_path_blocks(path_blocks: str) -> pd.DataFrame:
    df = pd.read_csv(path_blocks)
    int_columns = ["path_index", "block_id"]
    float_columns = ["cycle_count", "freq", "instr_count"]

    df[int_columns] = df[int_columns].astype(int)
    df[float_columns] = df[float_columns].astype(float)

    return df


def load_temperature_diff_stats(temp_diff: str) -> pd.DataFrame:
    # Should be single-part csv so can just use pandas
    df = pd.read_csv(temp_diff)
    df["block_id"] = df["block_id"].astype(int)
    float_columns = [
        "temp_max_etc",
        "temp_max_baseline",
        "temp_diff",
    ]
    df[float_columns] = df[float_columns].astype(float)

    return df


class PowerTraceRequestSpec:
    def __init__(
        self,
        block_id: int,
        voltage_level: float,
        voltage_levels: list,
        mcpat_input_folder: str | None,
        mcpat_output_folder: str,
        program_name: str,
        stats_df: pd.DataFrame,
        input_xml: str | None,
        frequency: float,
        config,
    ):
        self.block_id = block_id
        self.voltage_level = voltage_level
        self.voltage_levels = voltage_levels
        self.mcpat_input_folder = mcpat_input_folder
        self.mcpat_output_folder = mcpat_output_folder
        self.program_name = program_name
        self.stats_df = stats_df
        self.input_xml = input_xml
        self.frequency = frequency
        self.config = config

    # TODO: existence of this function breaks our procedural paradigm, but its convenient
    def change_to_other_config(
        self, block_id: int, voltage_level: float, frequency: float
    ) -> None:
        self.block_id = block_id
        self.voltage_level = voltage_level
        self.frequency = frequency


# TODO: "name" is actually name prefix
def mcpat_get_unit_stats(name: str, text: str) -> dict | None:
    # Note sometimes a colon, sometimes not
    pattern = rf"\s*{re.escape(name)}.*\n((?:.+\s*=\s*.+\n)+)"

    m = re.search(pattern, text, re.MULTILINE)

    if not m:
        return None

    block = m.group(1)
    fields = {}

    for key in [
        "Area",
        "Peak Dynamic",
        "Subthreshold Leakage",
        "Gate Leakage",
        "Runtime Dynamic",
    ]:
        fm = re.search(rf"\s+{key}\s*=\s*(\S+)\s.+\n", block, re.MULTILINE)
        if fm:
            # print(f"{fm.group(1)=}")
            fields[key] = float(fm.group(1))

    # print(f"{fields=}")

    return fields


def get_static_dynamic_power(unit_stats: dict, keys: list) -> float:
    total = 0.0

    for key in keys:
        if key not in unit_stats or unit_stats[key] is None:
            fatal(f"Missing stat {key} from unit stats {unit_stats}")

        stats = unit_stats[key]
        total += stats.get("Runtime Dynamic", 0)
        total += stats.get("Gate Leakage", 0)
        total += stats.get("Subthreshold Leakage", 0)

    return total


def mcpat_to_dict(mcpat_out_file: str) -> dict:
    text = ""

    with open(mcpat_out_file, "r") as f:
        text = f.read()

    unit_stats = {}

    for unit in MCPAT_DESIRED_UNITS:
        stats = mcpat_get_unit_stats(unit, text)
        unit_stats[unit] = stats

    return unit_stats


def load_folder_mcpat(folder_path: str, file_prefix: str = "") -> pd.DataFrame:
    folder_path = folder_path.rstrip(os.sep)
    last_directory_name = os.path.basename(folder_path)

    if file_prefix == "":
        file_prefix = last_directory_name

    if file_prefix == "":
        fatal(f"Bad folder path? File prefix was empty. {folder_path=}")

    pattern = re.compile(rf"{re.escape(file_prefix)}_idx(\d{{4}})_v(\d+)\.txt$")

    all_data = defaultdict(list)

    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)

        if not os.path.isfile(file_path):
            continue

        mcpat_unit_stats = mcpat_to_dict(file_path)

        for key in MCPAT_DESIRED_UNITS:
            mcpat_unit_stats[key] = get_static_dynamic_power(mcpat_unit_stats, [key])

        match = pattern.match(f)

        if not match:
            fatal(f"Unrecognised file name format: {f}, in path {file_path}")

        path_number = int(match.group(1))
        voltage_level = int(match.group(2))

        all_data["voltage"].append(voltage_level)

        # TODO: we don't know if these are paths or blocks, so we have to add both
        all_data["path_index"].append(path_number)
        all_data["block_id"].append(path_number)

        for key, power in mcpat_unit_stats.items():
            all_data[key].append(power)

    return pd.DataFrame(all_data)


def tei_get_voltage(temperature_celsius: float, target_frequency_ghz: float):
    # TODO: not too impactful most likely, but these constants are for 16nm!
    d_0 = -4.27
    d_1 = 0.0042
    d_2 = 0.0052
    d_3 = 10.6
    d_4 = -2.66
    # f = d_0x^2+d_1xt+d_2t+d_3x+d_4
    # f = d_0x^2+(d_1t+d_3)x+d_2t+d4
    # f = d_0x^2+b_1x+b_2

    b_1 = d_1 * temperature_celsius + d_3
    b_2 = d_2 * temperature_celsius + d_4 - target_frequency_ghz

    # Note we only want solutions on the left side of the peak of the curve
    #   so we only consider the + case
    # ax^2+bx+c
    a = d_0
    b = b_1
    c = b_2

    required_voltage = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)

    return required_voltage


def tei_get_frequency(temperature_celsius: float, voltage: float):
    d_0 = -4.27
    d_1 = 0.0042
    d_2 = 0.0052
    d_3 = 10.6
    d_4 = -2.66

    if math.isnan(temperature_celsius):
        error(f"Got NaN temperature in get_frequency, defaulting to 77")
        # TODO: default temperature from config
        temperature_celsius = 77

    info(f"Getting frequency, voltage: {voltage}, temp: {temperature_celsius}")

    # f_{max}=d_0 V_{dd}^2+d_1V_{dd}T+d_2T+d_3V_{dd}+d_4

    # TODO: can re-arrange into a nice quadratic

    return (
        d_0 * voltage * voltage
        + d_1 * voltage * temperature_celsius
        + d_2 * temperature_celsius
        + d_3 * voltage
        + d_4
    )


def round_to_precision(value: float, precision: float) -> float:
    return round(value / precision) * precision


def tei_select_frequency(
    config: configparser.ConfigParser, temperature_celsius: float, voltage: float
) -> float:
    frequency = tei_get_frequency(temperature_celsius, voltage)

    precision = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_FREQUENCY_PRECISION])
    maximum_frequency = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_FREQUENCY_LIMIT])

    if math.isnan(frequency):
        error("Received NaN frequency from tei_get_frequency")

        return float(config[MCPAT_CFG_MODULE_NAME][MCPAT_CLOCK_RATE_MHZ]) / 1000.0

    selected_frequency = round(frequency / precision) * precision
    selected_frequency = min(selected_frequency, maximum_frequency)

    return selected_frequency


def tei_select_voltage(
    config,
    temperature_celsius: float,
    target_frequency_ghz: float,
    voltage_levels: list,
) -> float:
    """
    Given config options, select an appropriate voltage for the target frequency, considering the effects of TEI
    """
    baseline_voltage = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_BASELINE_VOLTAGE])
    attempt_temperature_optim = bool(
        config[MCPAT_CFG_MODULE_NAME][MCPAT_ATTEMPT_TEMP_OPTIM]
    )
    low_temperature = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_TEMP_LOW])
    optim_temperature = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_TEMP_OPTIM])
    round_up = bool(config[MCPAT_CFG_MODULE_NAME][MCPAT_ROUND_UP])
    voltage_adjustment = float(
        config[MCPAT_CFG_MODULE_NAME][MCPAT_VOLTAGE_UPWARDS_ADJUSTMENT]
    )

    info(
        f"Grabbing voltage for temp {temperature_celsius}, at freq {target_frequency_ghz}, for levels: {voltage_levels}"
    )

    required_voltage = tei_get_voltage(temperature_celsius, target_frequency_ghz)
    temperature_kelvin = temperature_celsius + 273.15

    # Select up a voltage level
    if (
        attempt_temperature_optim
        and temperature_kelvin < low_temperature
        and temperature_kelvin < optim_temperature
    ):
        required_voltage += voltage_adjustment
        # Increase voltage, but don't take it above the baseline, so we still prioritise energy
        required_voltage = min(required_voltage, baseline_voltage)

    epsilon = 0.01

    if round_up:
        for v in voltage_levels:
            if v >= (required_voltage - epsilon):
                return v
    else:
        for v in reversed(voltage_levels):
            if v <= (required_voltage + epsilon):
                return v

    # We failed on either path, meaning required voltage was too high, or too low
    # If rounding up, return highest voltage we have
    # Required voltage was too low, return smallest voltage
    final_voltage = voltage_levels[-1] if round_up else voltage_levels[0]

    error("Couldn't select sufficient voltage to support frequency under temperature constraints")

    return final_voltage


def tei_select_vf_pairs(
    config: configparser.ConfigParser,
    temperature_celsius: float,
    voltage_levels: list,
    force_thermal_safety: bool,
    force_baseline_voltage: bool,
    allow_variable_frequency: bool,
) -> list:
    # TODO: could consider further vf pairs,  but adds more computation and
    # there's not a good case for raising voltage/frequency higher; tends to hinder energy efficiency further
    precision = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_FREQUENCY_PRECISION])

    vf_pairs = []

    default_frequency = (
        float(config[MCPAT_CFG_MODULE_NAME][MCPAT_CLOCK_RATE_MHZ]) / 1000.0
    )
    default_voltage = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_BASELINE_VOLTAGE])

    initial_voltage = tei_select_voltage(
        config, temperature_celsius, default_frequency, voltage_levels
    )

    if force_thermal_safety:
        return [(initial_voltage, default_frequency)]

    maximum_frequency = tei_select_frequency(
        config, temperature_celsius, initial_voltage
    )

    if force_baseline_voltage:
        initial_voltage = default_voltage

    vf_pairs.append((initial_voltage, default_frequency))

    if allow_variable_frequency:
        current_freq = float(config[MCPAT_CFG_MODULE_NAME][MCPAT_FREQUENCY_LIMIT]) 

        # Try all frequencies
        while current_freq > default_frequency:
            info(f"Selecting alternative frequency: {current_freq}")
            required_voltage = tei_select_voltage(config, temperature_celsius, current_freq, voltage_levels)
            vf_pairs.append((required_voltage, current_freq))
            current_freq -= precision

    return vf_pairs


def generate_mcpat_power_name(
    program_name: str, block_id: int, voltage_index: int, frequency: float
) -> str:
    return f"{program_name}_idx{block_id:04d}_v{voltage_index}_{frequency:.1f}Hz"


def get_voltage_index(voltage_levels: list, voltage_level: float) -> int:
    # Usually we expect voltage_level to be at one of the existing voltage levels
    # however in the case its off by a small epsilon value, this will ensure
    if len(voltage_levels) == 0:
        utils.fatal("Did not get any voltage levels when requesting index")

    max_index, _ = min(
        enumerate(voltage_levels),
        key=lambda p: abs(voltage_levels[p[0]] - voltage_level),
    )

    return max_index


def request_power_for_specification(
    request_spec: PowerTraceRequestSpec, expect_exists=False
):
    power = _request_power_trace_for_voltage(
        request_spec.block_id,
        request_spec.voltage_level,
        request_spec.voltage_levels,
        request_spec.mcpat_input_folder,
        request_spec.mcpat_output_folder,
        request_spec.program_name,
        request_spec.stats_df,
        request_spec.input_xml,
        request_spec.frequency,
        request_spec.config,
        expect_exists,
    )

    if power is None:
        fatal(
            f"Program {request_spec.program_name} was missing power for {request_spec.block_id} at {request_spec.voltage_level}V for frequency {request_spec.frequency}"
        )

    return power


def _request_power_trace_for_voltage(
    block_id: int,
    voltage_level: float,
    voltage_levels: list,
    mcpat_input_folder: str | None,
    mcpat_output_folder: str,
    program_name: str,
    stats_df: pd.DataFrame,
    input_xml: str | None,
    frequency: float,
    config: configparser.ConfigParser,
    expect_exists: bool,
):
    # TODO: relatively trivial to take temperature at this point too
    """
    1. check if the given power trace already exists
    2. or generate power trace ourself
    - use states file and metadata to generate McPAT input
    - run shell commands to generate mcpat output
    3. extract mpcat power trace from file
    4. once given power trace, extract to regular dataframe and return
    """

    file_name = generate_mcpat_power_name(
        program_name,
        block_id,
        get_voltage_index(voltage_levels, voltage_level),
        frequency,
    )
    info(f"Looking for file name: {file_name}")

    output_power_trace = f"./{mcpat_output_folder}/{file_name}.txt"
    input_stats_xml = f"./{mcpat_input_folder}/{file_name}.xml"

    path = Path(output_power_trace)

    if not path.is_file():
        if (
            expect_exists
            or input_xml == ""
            or mcpat_input_folder == ""
            or input_xml is None
            or mcpat_input_folder is None
        ):
            # TODO: just add __repr__ and __str__ to RequestSpec
            info(
                f"<RequestSpec: {block_id=} {voltage_level=} {voltage_levels=} {program_name=} {expect_exists=}>"
            )
            return None

        info(
            f"Generating mcpat file {output_power_trace}, {block_id=} {voltage_level=}, {frequency=}"
        )

        # TODO: believe originates here, sometimes stats_df is very wrong
        # some structure like {column: {0: value}, ...}
        create_mcpat_input_xml_etc(
            input_xml,
            input_stats_xml,
            (stats_df.loc[stats_df["block_id"] == block_id].iloc[0]).to_dict(),
            config,
            get_voltage_index(voltage_levels, voltage_level),
            frequency,
        )

        subprocess.run(
            ["./run_mcpat_specific.sh", f"{file_name}", f"{program_name}"],
            check=True,
        )

    # Returns dictionary with power values for this given basic block
    mcpat_dict = mcpat_to_dict(output_power_trace)

    return mcpat_dict


def get_distributed_voltage_levels(voltage_levels: list, num_values: int) -> list:
    """
    Returns the list of indices into voltage levels
    """
    if num_values == 0:
        fatal("Expected to select at least 1 voltage level")

    if num_values > len(voltage_levels):
        fatal("Got too many voltage levels to select")

    total = len(voltage_levels)

    if num_values == 1:
        # TODO: use list slice instead
        return [0]

    step = (total - 1) / (num_values - 1)
    indices = [math.floor(i * step) for i in range(num_values)]
    # Replace last of our indices with the largest voltage
    # ensures the range of values we generate always begins with minimum voltage, and ends with largest voltage
    indices[-1] = total - 1

    # Return the keys of the voltage levels
    return indices


# TODO: output of this isn't dictionary?
# TODO: shouldn't be used anyway, prefer load_multipart_csv
def load_csv_to_dict(path: str):
    data = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data.append(row)

    return data


def load_cfg(path: str):
    cfg = configparser.ConfigParser()
    cfg.read(path)

    return cfg


def load_voltage_levels_from_cfg(cfg) -> list:
    mcpat_data = cfg["mcpat"]

    i = 0

    voltage_levels = []

    prev_voltage = 0.0

    # arbitrary limit of 50 voltage levels
    for i in range(50):
        voltage_key = f"V{i}"

        if voltage_key not in mcpat_data:
            break

        value = float(mcpat_data[voltage_key])

        if value < prev_voltage:
            fatal("Voltage levels must be non-decreasing!")

        if value == prev_voltage:
            warn(
                "Omit voltage levels if unused, do not have same voltage value for multiple voltage levels"
            )
            break

        voltage_levels.append(value)

    return sorted(voltage_levels)


def load_program_heats(heat_table: str) -> pd.DataFrame:
    # Should be single-part csv so can just use pandas
    df = pd.read_csv(heat_table)
    df["block_id"] = df["block_id"].astype(int)
    float_columns = [
        "temp_mean",
        "temp_max",
        "temp_area_weighted_mean",
        "cycle_count",
    ]
    df[float_columns] = df[float_columns].astype(float)

    return df


## TODO: load floorplan


def load_voltage_frequency(vf_pairs: str) -> pd.DataFrame:
    df = pd.read_csv(vf_pairs)
    df["block_id"] = df["block_id"].astype(int)
    float_columns = ["voltage", "frequency"]
    df[float_columns] = df[float_columns].astype(float)

    return df


def load_program_heats_voltages(hv_table: str) -> pd.DataFrame:
    df = pd.read_csv(hv_table)
    df["block_id"] = df["block_id"].astype(int)
    float_columns = [
        "temp_mean",
        "temp_max",
        "temp_area_weighted_mean",
        "execution_cycles",
        "execution_time",
        "dvs_calling_count",
        "required_voltage_value",
        "obtained_frequency",
    ]
    df[float_columns] = df[float_columns].astype(float)

    return df


def load_program_temperature_change(temp_diff: str) -> pd.DataFrame:
    df = pd.read_csv(temp_diff)
    df["block_id"] = df["block_id"].astype(int)
    float_columns = [
        "temp_max_etc",
        "temp_max_baseline",
        "temp_diff",
    ]
    df[float_columns] = df[float_columns].astype(float)

    return df


def load_multipart_csv(path: str, delim=",") -> list:
    part = {}  # dict of arrays, one entry for each row, arranged like a dataframe
    data = []
    header = ""
    cols = []

    with open(path, newline="", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip("\n")

            if (not header) or (line == header and header):
                header = line
                data.append(copy.deepcopy(part))
                cols = header.split(delim)
                part = {x: [] for x in cols}
                continue

            fields = line.split(delim)

            assert len(cols) == len(
                fields
            ), f"Line: {line} was invalid, expected {len(cols)} columns, but got {len(fields)}"

            for col, field in zip(cols, fields):
                part[col].append(field)

    data.append(copy.deepcopy(part))

    return [i for i in data if len(i) > 0]


def load_adjacency_list_cfg(path: str, module_index: int) -> pd.DataFrame:
    csv_parts = load_multipart_csv(path)
    module_data = csv_parts[module_index]

    df = pd.DataFrame(module_data)
    df["start_block_id"] = df["start_block_id"].astype(int)
    df["exit_block_id"] = df["exit_block_id"].astype(int)
    df["branch_prob"] = df["branch_prob"].astype(float)
    df["start_path_index"] = df["start_path_index"].astype(int)
    df["end_path_index"] = df["end_path_index"].astype(int)
    df["is_start_entry"] = df["is_start_entry"].astype(int)

    return df


def load_voltage_levels(path: str) -> pd.DataFrame:
    csv_parts = load_multipart_csv(path)

    if len(csv_parts) == 0:
        fatal("Voltage levels were not initialised")

    df = pd.DataFrame(csv_parts[0])
    df["block_id"] = df["block_id"].astype(int)
    df["voltage_level"] = df["voltage_level"].astype(float)

    return df


def load_adjacency_list_dag(path: str, module_index: int) -> pd.DataFrame:
    csv_parts = load_multipart_csv(path)
    module_data = csv_parts[module_index]

    df = pd.DataFrame(module_data)
    df["start_comp"] = df["start_comp"].astype(int)
    df["end_comp"] = df["end_comp"].astype(int)

    return df


def load_topo_sort(path: str, module_index: int) -> list:
    csv_parts = load_multipart_csv(path)
    module_data = csv_parts[module_index]

    df = pd.DataFrame(module_data)
    df["comp_id"] = df["comp_id"].astype(int)
    df["comp_priority"] = df["comp_priority"].astype(int)

    comp_ids = df.sort_values("comp_priority", ascending=True)["comp_id"].tolist()

    return comp_ids


def load_block_additional(path: str, module_index: int) -> pd.DataFrame:
    csv_parts = load_multipart_csv(path)
    module_data = csv_parts[module_index]

    df = pd.DataFrame(module_data)
    df["block_id"] = df["block_id"].astype(int)
    df["comp_id"] = df["comp_id"].astype(int)
    df["execution_cycles"] = df["execution_cycles"].astype(float)

    return df


def get_block_ids(
    module_index: int, additional_data: str = "PerBlockAdditional.csv"
) -> list:
    """
    Returns the block ids given in PerBlockAdditional.csv
    """
    block_df = load_block_additional(additional_data, module_index)

    return block_df["block_id"].tolist()


def mcpat_to_hotspot_units(
    unit_stats: dict, flp_df: pd.DataFrame, use_residuals=True
) -> dict:
    # Unsure how to attribute most of these residuals, just doing the most impactful ones
    # Instruction Fetch Unit residual
    ifu_residual = get_static_dynamic_power(
        unit_stats, ["Instruction Fetch Unit"]
    ) - get_static_dynamic_power(
        unit_stats,
        [
            "Instruction Cache",
            "Branch Target Buffer",
            "Branch Predictor",
            "Instruction Buffer",
            "Instruction Decoder",
        ],
    )

    # Renaming Unit
    renaming_residual = get_static_dynamic_power(
        unit_stats, ["Renaming Unit"]
    ) - get_static_dynamic_power(
        unit_stats,
        [
            "Int Front End RAT",
            "FP Front End RAT",
            "Int Retire RAT",
            "FP Retire RAT",
            "FP Free List",
            "Free List",
        ],
    )
    # Load Store Unit
    lsu_residual = get_static_dynamic_power(
        unit_stats, ["Load Store Unit"]
    ) - get_static_dynamic_power(
        unit_stats,
        [
            "Data Cache",
            "LoadQ",
            "StoreQ",
        ],
    )

    # Memory Management
    mmu_residual = get_static_dynamic_power(
        unit_stats, ["Memory Management Unit"]
    ) - get_static_dynamic_power(
        unit_stats,
        [
            "Itlb",
            "Dtlb",
        ],
    )
    # Execution Unit
    execution_residual = get_static_dynamic_power(
        unit_stats, ["Execution Unit"]
    ) - get_static_dynamic_power(
        unit_stats,
        [
            "Register Files",
            "Instruction Scheduler",
            "Integer ALU",
            "Floating Point Unit",
            "Results Broadcast Bus",
        ],
    )
    execution_units = [
        "FPAdd_0",
        "FPAdd_1",
        "FPReg_0",
        "FPReg_1",
        "FPReg_2",
        "FPReg_3",
        "FPMul_0",
        "FPMul_1",
        "FPQ",
        "IntQ",
        "IntExec",
        "IntReg_0",
        "IntReg_1",
    ]

    execution_total_area = flp_df.loc[
        flp_df["unit"].isin(execution_units),
        "area",
    ].sum()

    # Should only be one entry, max is arbitrary
    area_fraction = (
        lambda unit_name, area: flp_df.loc[flp_df["unit"] == unit_name, "area"].max()
        / area
    )

    ifu_residual *= use_residuals
    renaming_residual *= use_residuals
    lsu_residual *= use_residuals
    mmu_residual *= use_residuals
    execution_residual *= use_residuals

    l2_power = get_static_dynamic_power(unit_stats, ["L2"])
    icache_power = get_static_dynamic_power(unit_stats, ["Instruction Cache"])
    dcache_power = get_static_dynamic_power(unit_stats, ["Data Cache"])
    bpred_power = get_static_dynamic_power(unit_stats, ["Branch Predictor"])
    dtb_power = get_static_dynamic_power(unit_stats, ["Dtlb"])
    fpu_power = get_static_dynamic_power(unit_stats, ["Floating Point Unit"])
    fp_rf_power = get_static_dynamic_power(unit_stats, ["Floating Point RF"])
    fp_map_power = get_static_dynamic_power(
        unit_stats, ["FP Front End RAT", "FP Retire RAT", "FP Free List"]
    )
    int_map_power = get_static_dynamic_power(
        unit_stats, ["Int Front End RAT", "Int Retire RAT", "Free List"]
    )
    intq_power = get_static_dynamic_power(unit_stats, ["Instruction Window"])
    int_rf_power = get_static_dynamic_power(unit_stats, ["Integer RF"])
    int_alu_power = get_static_dynamic_power(unit_stats, ["Integer ALU"])
    fpq_power = get_static_dynamic_power(unit_stats, ["FP Instruction Window"])
    lsq_power = get_static_dynamic_power(unit_stats, ["LoadQ", "StoreQ"])
    itb_power = get_static_dynamic_power(unit_stats, ["Itlb"])

    # Now map unit stats to hotspot
    # Assuming Alpha EV6 floorplan
    # TODO: Alpha EV6 seems to be different to the Alpha21364.xml we use as our baseline?
    # Note we don't have a direct mapping, so we take some averages
    # TODO: mathematical correctness of averaging? shouldn't we consider it to just be one component with less heat spread?
    # TODO: unsure about which McPAT components to use
    hotspot_mapping = {
        "L2_left": l2_power / 3.0,
        "L2": l2_power / 3.0,
        "L2_right": l2_power / 3.0,
        "Icache": icache_power,
        "Dcache": dcache_power,
        "Bpred_0": bpred_power / 3.0,
        "Bpred_1": bpred_power / 3.0,
        "Bpred_2": bpred_power / 3.0,
        "DTB_0": dtb_power / 3.0,
        "DTB_1": dtb_power / 3.0,
        "DTB_2": dtb_power / 3.0,
        # NOTE: averaging these ones over the Add and Mul units
        "FPAdd_0": fpu_power / 4.0
        + area_fraction("FPAdd_0", execution_total_area) * execution_residual,
        "FPAdd_1": fpu_power / 4.0
        + area_fraction("FPAdd_1", execution_total_area) * execution_residual,
        "FPReg_0": fp_rf_power / 4.0
        + area_fraction("FPReg_0", execution_total_area) * execution_residual,
        "FPReg_1": fp_rf_power / 4.0
        + area_fraction("FPReg_1", execution_total_area) * execution_residual,
        "FPReg_2": fp_rf_power / 4.0
        + area_fraction("FPReg_2", execution_total_area) * execution_residual,
        "FPReg_3": fp_rf_power / 4.0
        + area_fraction("FPReg_3", execution_total_area) * execution_residual,
        "FPMul_0": fpu_power / 4.0
        + area_fraction("FPMul_0", execution_total_area) * execution_residual,
        "FPMul_1": fpu_power / 4.0
        + area_fraction("FPMul_1", execution_total_area) * execution_residual,
        "FPMap_0": fp_map_power / 2.0,
        "FPMap_1": fp_map_power / 2.0,
        "IntMap": int_map_power,
        "IntQ": intq_power
        + area_fraction("IntQ", execution_total_area) * execution_residual,
        "IntReg_0": int_rf_power / 2.0
        + area_fraction("IntReg_0", execution_total_area) * execution_residual,
        "IntReg_1": int_rf_power / 2.0
        + area_fraction("IntReg_1", execution_total_area) * execution_residual,
        "IntExec": int_alu_power
        + area_fraction("IntExec", execution_total_area) * execution_residual,
        "FPQ": fpq_power
        + area_fraction("FPQ", execution_total_area) * execution_residual,
        "LdStQ": lsq_power,
        "ITB_0": itb_power / 2.0,
        "ITB_1": itb_power / 2.0,
    }

    return hotspot_mapping


def set_voltages(voltage_file: str, voltages: list, block_ids: list):
    # TODO: validation on voltage levels
    if len(voltages) != len(block_ids):
        fatal("Expected list of voltages and block ids to be same length")

    voltage_df = pd.DataFrame()
    voltage_df["voltage_level"] = voltages
    voltage_df["block_id"] = block_ids

    voltage_df.to_csv(voltage_file, index=False)


def init_voltages(voltage_file: str, voltage_level: int, block_ids: list):
    set_voltages(voltage_file, [voltage_level for _ in block_ids], block_ids)


def change_xml_property(
    tree, component_path: str, param_or_stat: str, name: str, new_value: str
):
    xpath = "."

    for id in component_path.split("/"):
        xpath += f"/component[@id='{id}']"
        # TODO: crate subelements here if they don't exist
        #   otherwise adding new components might fail

    parent_path = xpath
    xpath += f"/{param_or_stat}[@name='{name}']"

    element = tree.find(xpath)

    if element is None:
        warn(f"Element {component_path}/{name} was not found, adding")
        parent = tree.find(parent_path)

        if parent is None:
            fatal("TODO: we should create the tree of parents before the element..")

        element = ET.SubElement(
            parent, param_or_stat, attrib={"name": name, "value": new_value}
        )

    element.set("value", new_value)


def get_stats_df_mbbs(csv_path: str, module_index: int) -> pd.DataFrame:
    data = load_multipart_csv(csv_path)

    mbbs = data[module_index]
    df = pd.DataFrame(mbbs)

    # TODO: set required columns as a global
    cols = [
        CYCLE_COUNT,
        INSTR_COUNT,
        INT_INSTR_COUNT,
        FLOAT_INSTR_COUNT,
        BRANCH_INSTR_COUNT,
        LOADS,
        STORES,
        INT_REGFILE_READS,
        INT_REGFILE_WRITES,
        FLOAT_REGFILE_READS,
        FLOAT_REGFILE_WRITES,
        FUNCTION_CALLS,
        CONTEXT_SWITCHES,
        MUL_ACCESS,
        FP_ACCESS,
        IALU_ACCESS,
    ]

    # NOTE: I64 is a little close to being too small
    # Attempt to just cast everything to float
    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").round().astype("Int64")
    )

    if not set(cols).issubset(df.columns):
        fatal(
            f"Input {csv_path} to mbb load of module {module_index} was missing required columns"
        )

    # df[cols] = df[cols].astype(float).round().astype("Int64")
    df = stats_df_estimate_missing_cols(df)
    df["block_id"] = df["block_id"].astype(int)

    return df


def get_stats_df_subgraphs(csv_path: str, module_index: int) -> [pd.DataFrame]:
    data = load_multipart_csv(csv_path)

    mbbs = data[module_index]
    df = pd.DataFrame(mbbs)

    cols = [
        CYCLE_COUNT,
        INSTR_COUNT,
        INT_INSTR_COUNT,
        FLOAT_INSTR_COUNT,
        BRANCH_INSTR_COUNT,
        LOADS,
        STORES,
        INT_REGFILE_READS,
        INT_REGFILE_WRITES,
        FLOAT_REGFILE_READS,
        FLOAT_REGFILE_WRITES,
        FUNCTION_CALLS,
        CONTEXT_SWITCHES,
        MUL_ACCESS,
        FP_ACCESS,
        IALU_ACCESS,
    ]

    # NOTE: I64 is a little close to being too small

    # Attempt to just cast everything to float
    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").round().astype("Int64")
    )

    if not set(cols).issubset(df.columns):
        fatal(
            f"Input {csv_path} to path load of module {module_index} was missing required columns"
        )

    df[PATH_INDEX] = df[PATH_INDEX].astype("Int64")
    df[IS_ENTRY] = df[IS_ENTRY].astype(bool)
    df[IS_EXIT] = df[IS_EXIT].astype(bool)
    # TODO: very bad, but easiest solution
    df["block_id"] = df[PATH_INDEX].copy()

    # This doesn't include the estimated columns, because we haven't added estimated columns
    other_cols = [c for c in df.columns if c not in cols]

    dfs = []
    for path, group in df.groupby(PATH_INDEX):
        summed = group[cols].sum()

        summed[PATH_INDEX] = path
        group_df = summed.to_frame().T
        group_df = stats_df_estimate_missing_cols(group_df)

        # Adding back in useful columns
        for c in other_cols:
            group_df[c] = group[c].iloc[0]

        dfs.append(group_df)

    # TODO: additional inputs/outputs for CFG and per-block stats here?
    # - at least the per-block entry/exit can be taken from here
    # - CFG we need additional input for so another function can handle that likely

    return dfs


def get_stats_df_gem5_run(input_path: str) -> pd.DataFrame:
    block_pattern = re.compile(
        r"-+\s*Begin Simulation Statistics\s*-+(.*?)"
        r"-+\s*End Simulation Statistics\s*-+",
        re.DOTALL | re.IGNORECASE
    )

    # generic stat line: <name><whitespace><value><whitespace><anything>
    stat_pattern = re.compile(
        r"""
        ^(?P<name>\S+)\s+
        (?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)
        \s+.*$
        """,
        re.MULTILINE | re.VERBOSE
    )

    with open(input_path, "r") as f:
        text = f.read()

    rows = []

    for i, block in enumerate(block_pattern.findall(text)):
        stats = {"block": i}
        for match in stat_pattern.finditer(block):
            name = match.group("name")
            value = float(match.group("value"))
            stats[name] = value

        renamed_stats = {}

        # Mapping gem5 data columns to our columns
        # Some of these are additional stats from our DVFS simulator
        renamed_stats[DVFS_CALL_COUNT] = int(stats.get("board.processor.cores.core.dvfsCallCount", 0))
        renamed_stats[AVERAGE_VOLTAGE] = float(stats.get("board.processor.cores.core.averageVoltage", 0))
        renamed_stats[AVERAGE_FREQUENCY] = float(stats.get("board.processor.cores.core.averageFrequency", 0))
        renamed_stats[IPC] = float(stats.get("board.processor.cores.core.ipc", 0))

        renamed_stats[CYCLE_COUNT] = int(stats.get("board.processor.cores.core.numCycles", 0))
        renamed_stats[IDLE_CYCLES] = int(stats.get("board.processor.cores.core.idleCycles", 0))
        renamed_stats[INSTR_COUNT] = int(stats.get("board.processor.cores.core.commitStats0.numInsts", 0))
        renamed_stats[INT_INSTR_COUNT] = int(
            stats.get("board.processor.cores.core.commitStats0.numIntInsts", 0)
        )
        renamed_stats[FLOAT_INSTR_COUNT] = int(
            stats.get("board.processor.cores.core.commitStats0.numFpInsts", 0)
        )
        renamed_stats[BRANCH_INSTR_COUNT] = int(
            stats.get("board.processor.cores.core.executeStats0.numBranches", 0)
        )
        renamed_stats[BRANCH_MISPREDICTIONS] = int(
            stats.get("board.processor.cores.core.commit.branchMispredicts", 0)
        )
        renamed_stats[LOADS] = int(stats.get("board.processor.cores.core.commitStats0.numLoadInsts", 0))
        renamed_stats[STORES] = int(stats.get("board.processor.cores.core.commitStats0.numStoreInsts", 0))
        renamed_stats[ROB_READS] = int(stats.get("board.processor.cores.core.rob.reads", 0))
        renamed_stats[ROB_WRITES] = int(stats.get("board.processor.cores.core.rob.writes", 0))
        renamed_stats[RENAME_READS] = int(stats.get("board.processor.cores.core.rename.lookups", 0))
        renamed_stats[RENAME_WRITES] = int(stats.get("board.processor.cores.core.rename.renamedOperands", 0))
        renamed_stats[FP_RENAME_READS] = int(stats.get("board.processor.cores.core.rename.fpLookups", 0))
        renamed_stats[INST_WINDOW_READS] = int(
            stats.get("board.processor.cores.core.intInstQueueReads", 0)
        ) + int(stats.get("board.processor.cores.core.fpInstQueueReads", 0))
        renamed_stats[INST_WINDOW_WRITES] = int(
            stats.get("board.processor.cores.core.intInstQueueWrites", 0)
        ) + int(stats.get("board.processor.cores.core.fpInstQueueWrites", 0))
        renamed_stats[INST_WINDOW_WAKEUP_ACCESSES] = int(
            stats.get("board.processor.cores.core.intInstQueueWakeupAccesses", 0)
        ) + int(stats.get("board.processor.cores.core.fpInstQueueWakeupAccesses", 0))
        renamed_stats[FP_INST_WINDOW_READS] = int(stats.get("board.processor.cores.core.fpInstQueueReads", 0))
        renamed_stats[FP_INST_WINDOW_WRITES] = int(
            stats.get("board.processor.cores.core.fpInstQueueWrites", 0)
        )
        renamed_stats[FP_INST_WINDOW_WAKEUP_ACCESSES] = int(
            stats.get("board.processor.cores.core.fpInstQueueWakeupAccesses", 0)
        )

        renamed_stats[INT_REGFILE_READS] = int(
            stats.get("board.processor.cores.core.executeStats0.numIntRegReads", 0)
        )
        renamed_stats[INT_REGFILE_WRITES] = int(
            stats.get("board.processor.cores.core.executeStats0.numIntRegWrites", 0)
        )
        renamed_stats[FLOAT_REGFILE_READS] = int(
            stats.get("board.processor.cores.core.executeStats0.numFpRegReads", 0)
        )
        renamed_stats[FLOAT_REGFILE_WRITES] = int(
            stats.get("board.processor.cores.core.executeStats0.numFpRegWrites", 0)
        )
        renamed_stats[FUNCTION_CALLS] = int(stats.get("board.processor.cores.core.commit.functionCalls", 0))
        renamed_stats[CONTEXT_SWITCHES] = int(
            stats.get("board.processor.cores.core.commitStats0.committedControl::IsReturn", 0)
        ) + int(stats.get("board.processor.cores.core.commitStats0.committedControl::IsCall", 0))

        # TODO: is this wrong?
        renamed_stats[MUL_ACCESS] = (
            int(stats.get("board.processor.cores.core.commitStats0.committedInstType::IntMult", 0))
            + int(
                stats.get("board.processor.cores.core.commitStats0.committedInstType::FloatMult", 0)
            )
            + int(
                stats.get("board.processor.cores.core.commitStats0.committedInstType::SimdMult", 0)
            )
            + int(
                stats.get(
                    "board.processor.cores.core.commitStats0.committedInstType::SimdFloatMult", 0
                    )
            )
        )

        renamed_stats[FP_ACCESS] = int(stats.get("board.processor.cores.core.fpAluAccesses", 0))
        renamed_stats[IALU_ACCESS] = int(stats.get("board.processor.cores.core.intAluAccesses", 0))

        renamed_stats[CDB_ALU_ACCESSES] = renamed_stats[IALU_ACCESS]
        renamed_stats[CDB_FP_ACCESSES] = renamed_stats[FP_ACCESS]
        renamed_stats[CDB_MUL_ACCESSES] = renamed_stats[MUL_ACCESS]

        renamed_stats[BTB_READS] = int(stats.get("board.processor.cores.core.branchPred.BTBLookups", 0))
        renamed_stats[BTB_WRITES] = int(stats.get("board.processor.cores.core.branchPred.BTBUpdates", 0))

        renamed_stats[COMMITTED_INSTR] = int(stats.get("board.processor.cores.core.commitStats0.numInsts", 0))
        renamed_stats[COMMITTED_INT_INSTR] = int(
            stats.get("board.processor.cores.core.commitStats0.numIntInsts", 0)
        )
        renamed_stats[COMMITTED_FLOAT_INSTR] = int(
            stats.get("board.processor.cores.core.commitStats0.numFpInsts", 0)
        )

        renamed_stats[FP_RENAME_WRITES] = int(stats.get("board.processor.cores.core.rename.fpLookups", 0)) / 2

        renamed_stats[BUSY_CYCLES] = renamed_stats[CYCLE_COUNT] - renamed_stats[IDLE_CYCLES]

        renamed_stats[L1D_HITS] = int(stats.get("board.cache_hierarchy.l1dcaches.ReadReq.hits::total", 0))
        renamed_stats[L1D_MISS] = int(stats.get("board.cache_hierarchy.l1dcaches.ReadReq.misses::total", 0))

        renamed_stats[L1I_HITS] = int(stats.get("board.cache_hierarchy.l1icaches.ReadReq.hits::total", 0))
        renamed_stats[L1I_MISS] = int(stats.get("board.cache_hierarchy.l1icaches.ReadReq.misses::total", 0))

        renamed_stats[L2_HITS] = int(
            stats.get("board.cache_hierarchy.l2caches.ReadExReq.hits::total", 0)
        ) + int(stats.get("board.cache_hierarchy.l2caches.ReadSharedReq.hits::total", 0))

        renamed_stats[L2_MISS] = int(
            stats.get("board.cache_hierarchy.l2caches.ReadExReq.misses::total", 0)
        ) + int(stats.get("board.cache_hierarchy.l2caches.ReadSharedReq.misses::total", 0))

        rows.append(renamed_stats)

    # TODO: does this work? we're creating a df off a dict of scalar values, probably will need lists or this will auto-convert to a series or error
    return pd.DataFrame(rows)


def stats_df_estimate_missing_cols(df: pd.DataFrame):
    """
    Estimates some less important stats from required stats

    Note some of these estimates are very crude
    """
    df[IDLE_CYCLES] = 0
    df[BUSY_CYCLES] = df[CYCLE_COUNT]
    df[BRANCH_MISPREDICTIONS] = 0
    df[COMMITTED_INSTR] = df[INSTR_COUNT]
    df[COMMITTED_INT_INSTR] = df[INT_INSTR_COUNT]
    df[COMMITTED_FLOAT_INSTR] = df[FLOAT_INSTR_COUNT]
    df[ROB_READS] = df[INSTR_COUNT]
    df[ROB_WRITES] = df[INSTR_COUNT]
    df[RENAME_READS] = df[INSTR_COUNT] * 2
    df[RENAME_WRITES] = df[INSTR_COUNT]
    df[FP_RENAME_READS] = df[FLOAT_INSTR_COUNT] * 2
    df[FP_RENAME_WRITES] = df[FLOAT_INSTR_COUNT]
    df[INST_WINDOW_WRITES] = df[INSTR_COUNT]
    df[INST_WINDOW_READS] = df[INSTR_COUNT]
    df[INST_WINDOW_WAKEUP_ACCESSES] = df[INST_WINDOW_WRITES] + df[INST_WINDOW_READS]
    df[FP_INST_WINDOW_READS] = df[FLOAT_INSTR_COUNT]
    df[FP_INST_WINDOW_WRITES] = df[FLOAT_INSTR_COUNT]
    df[FP_INST_WINDOW_WAKEUP_ACCESSES] = (
        df[FP_INST_WINDOW_READS] + df[FP_INST_WINDOW_WRITES]
    )
    df[CDB_MUL_ACCESSES] = df[MUL_ACCESS]
    df[CDB_ALU_ACCESSES] = df[IALU_ACCESS]
    df[CDB_FP_ACCESSES] = df[FP_ACCESS]
    df[BTB_READS] = df[INSTR_COUNT]
    df[BTB_WRITES] = 0

    return df


def create_mcpat_input_xml_etc(
    input_path: str,
    output_path: str,
    input_stats: dict,
    input_cfg: configparser.ConfigParser,
    voltage_level_id: int,
    frequency: float,
) -> None:
    voltage_levels = load_voltage_levels_from_cfg(input_cfg)
    vdd = f"{voltage_levels[voltage_level_id]:.2f}"

    create_mcpat_input_xml(input_path, output_path, input_stats, input_cfg, vdd, frequency * 1.0e3) 


def create_mcpat_input_xml(input_path: str, output_path: str, input_stats: dict, input_cfg: configparser.ConfigParser, vdd: float, frequency_mhz: float) -> None:
    tree = ET.parse(input_path)

    frequency = frequency_mhz

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mcpat_config = input_cfg[MCPAT_CFG_MODULE_NAME]

    change_xml_property(
        tree, "system", "param", "temperature", str(mcpat_config[MCPAT_TEMP])
    )
    change_xml_property(
        tree, "system", "param", "core_tech_node", str(mcpat_config[MCPAT_NODE_SIZE])
    )
    change_xml_property(
        tree,
        "system",
        "param",
        "target_core_clockrate",
        str(frequency),
    )
    change_xml_property(tree, "system/system.core0", "param", "vdd", str(vdd))

    change_xml_property(
        tree, "system", "stat", "total_cycles", str(input_stats[CYCLE_COUNT])
    )
    change_xml_property(
        tree, "system", "stat", "busy_cycles", str(input_stats[BUSY_CYCLES])
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "param",
        "clock_rate",
        str(frequency),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "total_instructions",
        str(input_stats[INSTR_COUNT]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "int_instructions",
        str(input_stats[INT_INSTR_COUNT]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_instructions",
        str(input_stats[FLOAT_INSTR_COUNT]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "branch_instructions",
        str(input_stats[BRANCH_INSTR_COUNT]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "branch_mispredictions",
        str(input_stats[BRANCH_MISPREDICTIONS]),
    )  # TODO: assume some % of branches miss
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "load_instructions",
        str(input_stats[LOADS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "store_instructions",
        str(input_stats[STORES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "committed_instructions",
        str(input_stats[COMMITTED_INSTR]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "committed_int_instructions",
        str(input_stats[COMMITTED_INT_INSTR]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "committed_fp_instructions",
        str(input_stats[COMMITTED_FLOAT_INSTR]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "total_cycles",
        str(input_stats[CYCLE_COUNT]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "idle_cycles",
        str(input_stats[IDLE_CYCLES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "busy_cycles",
        str(input_stats[BUSY_CYCLES]),
    )
    change_xml_property(
        tree, "system/system.core0", "stat", "ROB_reads", str(input_stats[ROB_READS])
    )
    change_xml_property(
        tree, "system/system.core0", "stat", "ROB_writes", str(input_stats[ROB_WRITES])
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "rename_reads",
        str(input_stats[RENAME_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "rename_writes",
        str(input_stats[RENAME_WRITES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_rename_reads",
        str(input_stats[FP_RENAME_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_rename_writes",
        str(input_stats[FP_RENAME_WRITES]),
    )

    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "inst_window_reads",
        str(input_stats[INST_WINDOW_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "inst_window_writes",
        str(input_stats[INST_WINDOW_WRITES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "inst_window_wakeup_accesses",
        str(input_stats[INST_WINDOW_WAKEUP_ACCESSES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_inst_window_reads",
        str(input_stats[FP_INST_WINDOW_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_inst_window_writes",
        str(input_stats[FP_INST_WINDOW_WRITES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "fp_inst_window_wakeup_accesses",
        str(input_stats[FP_INST_WINDOW_WAKEUP_ACCESSES]),
    )

    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "int_regfile_reads",
        str(input_stats[INT_REGFILE_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "float_regfile_reads",
        str(input_stats[FLOAT_REGFILE_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "int_regfile_writes",
        str(input_stats[INT_REGFILE_WRITES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "float_regfile_writes",
        str(input_stats[FLOAT_REGFILE_WRITES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "function_calls",
        str(input_stats[FUNCTION_CALLS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "context_switches",
        str(input_stats[CONTEXT_SWITCHES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "ialu_accesses",
        str(input_stats[IALU_ACCESS]),
    )
    change_xml_property(
        tree, "system/system.core0", "stat", "fpu_accesses", str(input_stats[FP_ACCESS])
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "mul_accesses",
        str(input_stats[MUL_ACCESS]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "cdb_alu_accesses",
        str(input_stats[CDB_ALU_ACCESSES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "cdb_fpu_accesses",
        str(input_stats[CDB_FP_ACCESSES]),
    )
    change_xml_property(
        tree,
        "system/system.core0",
        "stat",
        "cdb_mul_accesses",
        str(input_stats[CDB_MUL_ACCESSES]),
    )

    # TODO: missing stuff???
    change_xml_property(
        tree,
        "system/system.core0/system.core0.itlb",
        "stat",
        "total_accesses",
        str(input_stats[INSTR_COUNT] // 2),
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.core0/system.core0.itlb", "stat", "total_misses", str(4)
    )  # TODO: some small default
    change_xml_property(
        tree, "system/system.core0/system.core0.itlb", "stat", "conflicts", str(0)
    )

    change_xml_property(
        tree,
        "system/system.core0/system.core0.icache",
        "stat",
        "read_accesses",
        str(input_stats[INSTR_COUNT] // 2),
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.core0/system.core0.icache", "stat", "read_misses", str(0)
    )  # TODO: some small default
    change_xml_property(
        tree, "system/system.core0/system.core0.icache", "stat", "conflicts", str(0)
    )

    change_xml_property(
        tree,
        "system/system.core0/system.core0.dtlb",
        "stat",
        "total_accesses",
        str(input_stats[INSTR_COUNT]),
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.core0/system.core0.dtlb", "stat", "total_misses", str(0)
    )

    change_xml_property(
        tree,
        "system/system.core0/system.core0.dcache",
        "stat",
        "read_accesses",
        str(input_stats[INSTR_COUNT] * 2),
    )  # TODO: unsure
    change_xml_property(
        tree,
        "system/system.core0/system.core0.dcache",
        "stat",
        "write_accesses",
        str(0),
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.core0/system.core0.dcache", "stat", "read_misses", str(0)
    )
    change_xml_property(
        tree, "system/system.core0/system.core0.dcache", "stat", "write_misses", str(0)
    )

    change_xml_property(
        tree,
        "system/system.core0/system.core0.BTB",
        "stat",
        "read_accesses",
        str(input_stats[BTB_READS]),
    )
    change_xml_property(
        tree,
        "system/system.core0/system.core0.BTB",
        "stat",
        "write_accesses",
        str(input_stats[BTB_WRITES]),
    )

    change_xml_property(
        tree,
        "system/system.L1Directory0",
        "param",
        "clockrate",
        str(frequency),
    )
    change_xml_property(
        tree,
        "system/system.L1Directory0",
        "stat",
        "read_accesses",
        str(input_stats[INSTR_COUNT] * 2),
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.L1Directory0", "stat", "write_accesses", str(0)
    )  # TODO: unsure
    change_xml_property(
        tree, "system/system.L1Directory0", "stat", "read_misses", str(0)
    )
    change_xml_property(
        tree, "system/system.L1Directory0", "stat", "write_misses", str(0)
    )
    change_xml_property(tree, "system/system.L1Directory0", "stat", "conflicts", str(0))

    # TODO: all unsure
    change_xml_property(
        tree,
        "system/system.L2Directory0",
        "param",
        "clockrate",
        str(frequency),
    )
    change_xml_property(
        tree, "system/system.L2Directory0", "stat", "read_accesses", str(0)
    )
    change_xml_property(
        tree, "system/system.L2Directory0", "stat", "write_accesses", str(0)
    )
    change_xml_property(
        tree, "system/system.L2Directory0", "stat", "read_misses", str(0)
    )
    change_xml_property(
        tree, "system/system.L2Directory0", "stat", "write_misses", str(0)
    )

    change_xml_property(
        tree,
        "system/system.L20",
        "param",
        "clockrate",
        str(frequency),
    )
    change_xml_property(tree, "system/system.L20", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_misses", str(0))

    change_xml_property(tree, "system/system.L30", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_misses", str(0))

    change_xml_property(
        tree,
        "system/system.NoC0",
        "stat",
        "total_accesses",
        str(input_stats[INSTR_COUNT] // 4),
    )  # TODO: unsure
    change_xml_property(
        tree,
        "system/system.NoC0",
        "param",
        "clockrate",
        str(frequency),
    )

    change_xml_property(
        tree,
        "system/system.mc",
        "stat",
        "memory_accesses",
        str(input_stats[INSTR_COUNT] // 10),
    )  # TODO: unsure
    change_xml_property(
        tree,
        "system/system.mc",
        "stat",
        "memory_reads",
        str(input_stats[INSTR_COUNT] // 20),
    )
    change_xml_property(
        tree,
        "system/system.mc",
        "stat",
        "memory_writes",
        str(input_stats[INSTR_COUNT] // 20),
    )

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# Creating this custom stat format isn't the most useful anymore since we don't intend
#   to profile any gem5 or external analysis often
#   But to avoid redesign we'll keep
def load_arbitrary_stat_file(
    path: str, module_index: int = 0, path_index: int = -1, take_sum: bool = False
) -> pd.DataFrame:
    filename = os.path.basename(path)

    stats = None

    if filename == "PathBlocks.csv":
        all_stats = get_stats_df_subgraphs(path, module_index)

        if path_index == -1 and take_sum:
            fatal(
                "Cannot sum over subgraphs, use MBB stats or select a specific subgraph"
            )

        if path_index == -1:
            stats = pd.concat(all_stats, ignore_index=True)

        else:
            for stat_df in all_stats:
                if len(stat_df) == 1 and stat_df.iloc[0][PATH_INDEX] == path_index:
                    stats = stat_df
                    break

    elif filename == "MBB_stats.csv":
        all_stats = get_stats_df_mbbs(path, module_index)

        if path_index == -1 and take_sum:
            numerical_sums = all_stats.select_dtypes(include="number").sum()
            # Get first of all other columns
            other = all_stats.drop(columns=numerical_sums.index).iloc[0]
            stats = numerical_sums.combine_first(other)
            # Stats isn't strictly a dataframe at this point, but it works fine anyway
            stats = stats.to_frame().T
        elif path_index == -1:
            # TODO: copy probably redundant
            stats = all_stats.copy()
        else:
            stats = all_stats[path_index]
    # Assuming gem5 output
    elif filename == "stats.txt":
        stats = get_stats_df_gem5_run(path)
    elif "_STD.csv" in filename:
        # TODO: warn on this path? or just check this path is valid
        stats = load_standard_stat_file(path)
    else:
        fatal(f"Unknown stat file {filename}")

    if stats is None:
        fatal("Failed to load stat file")

    return stats


def create_standard_stat_file(
    path: str,
    output_name: str,
    module_index: int = 0,
    path_index: int = -1,
    take_sum: bool = False,
) -> None:
    stat_df = load_arbitrary_stat_file(path, module_index, path_index, take_sum)
    stat_df.to_csv(f"{output_name}_STD.csv", index=False)


def load_standard_stat_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").round().astype("Int64")
    )

    return df

def create_unified_program_stats(efficiency_stats: dict, output_csv: str) -> pd.DataFrame:
    df_dict = {
        "program_name": [],
        "edp_percent": [],
        "energy_percent": [],
        "ips_percent": [],
    }

    for program_name, program_data in efficiency_stats.items():
        df_dict["program_name"].append(program_name)
        df_dict["edp_percent"].append(-program_data["edp_potential"])
        df_dict["energy_percent"].append(-program_data["energy_potential"])
        df_dict["ips_percent"].append(program_data["ips_potential"])

    df = pd.DataFrame(df_dict)

    df.to_csv(output_csv, index=False)

    return df
