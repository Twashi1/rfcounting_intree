import csv
import xml.etree.ElementTree as ET
import argparse
import re
import copy
# not really required, but useful for some csv processing
import pandas as pd
import configparser
import os

MCPAT_CFG_MODULE_NAME = "mcpat"
MCPAT_CLOCK_RATE = "CLOCK_RATE"
MCPAT_TEMP = "TEMP"
MCPAT_NODE_SIZE = "NODE_SIZE"
MCPAT_VOLTAGE_LOW = "VOLTAGE_LOW"
MCPAT_VOLTAGE_MED = "VOLTAGE_MED"
MCPAT_VOLTAGE_HIGH = "VOLTAGE_HIGH"

# NOTE: should be an enum but it'd be difficult to use as keys
MODULE_NAME = "module_name"
FUNCTION_NAME = "function_name"
BLOCK_NAME = "block_name"

PATH_INDEX = "path_index"
IS_ENTRY = "is_entry"
IS_EXIT = "is_exit"

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

def load_multipart_csv(path: str, delim=",") -> list:
    part = {} # dict of arrays, one entry for each row, arranged like a dataframe
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

            assert len(cols) == len(fields), f"Line: {line} was invalid, expected {len(cols)} columns, but got {len(fields)}"

            for col, field in zip(cols, fields):
                part[col].append(field) 
    
    data.append(copy.deepcopy(part))

    return [i for i in data if len(i) > 0]

def change_xml_property(tree, component_path: str, param_or_stat: str, name: str, new_value: str):
    xpath = "."

    for id in component_path.split("/"):
        xpath += f"/component[@id='{id}']"
        # TODO: crate subelements here if they don't exist
        #   otherwise adding new components might fail

    parent_path = xpath
    xpath += f"/{param_or_stat}[@name='{name}']"

    element = tree.find(xpath)

    if element is None:
        print(f"[WARN] Element {component_path}/{name} was not found, adding")
        parent = tree.find(parent_path)

        if parent is None:
            raise NotImplementedError("TODO: we should create the tree of parents before the element..")

        element = ET.SubElement(parent, param_or_stat, attrib={"name": name, "value": new_value})

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
        IALU_ACCESS
    ]

    # NOTE: I64 is a little close to being too small
    # Attempt to just cast everything to float
    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").round().astype("Int64")
    )

    if not set(cols).issubset(df.columns):
        raise ValueError(f"Input {csv_path} to mbb load of module {module_index} was missing required columns")

    # df[cols] = df[cols].astype(float).round().astype("Int64")
    df = stats_df_estimate_missing_cols(df)

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
        IALU_ACCESS
    ]

    # NOTE: I64 is a little close to being too small
    
    # Attempt to just cast everything to float
    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").round().astype("Int64")
    )

    if not set(cols).issubset(df.columns):
        print(df.columns, cols)
        raise ValueError(f"Input {csv_path} to path load of module {module_index} was missing required columns")

    df[PATH_INDEX] = df[PATH_INDEX].astype("Int64")
    df[IS_ENTRY] = df[IS_ENTRY].astype(bool)
    df[IS_EXIT] = df[IS_EXIT].astype(bool)

    df = stats_df_estimate_missing_cols(df)

    # Accumulate stats for each given path index
    # then re-estimate missing stats and add to list
    filtered = df[cols + [PATH_INDEX]]

    dfs = []
    for path, group in filtered.groupby(PATH_INDEX):
        summed = group[cols].sum()
        summed[PATH_INDEX] = path
        group_df = summed.to_frame().T
        group_df = stats_df_estimate_missing_cols(group_df)
        # TODO: add back in useful columns to group_df
        dfs.append(group_df)

    return dfs

def get_stats_df_calling_points():
    raise NotImplementedError()

def get_stats_df_gem5_run(input_path: str) -> pd.DataFrame:
    df = {}
    data = {}

    with open(input_path, "r") as f:
        # match non-whitespace, whitespace, non-whitespace
        # so name, whitespace, value
        pattern = re.compile(r"^(\S+)\s+(\S+)")
        
        for line in f.readlines():
            # ignore --- begin/end simulation
            if line.startswith("----"):
                continue

            # empty line
            if not line:
                continue

            m = pattern.match(line)

            if not m:
                continue

            key, val = m.groups()
            data[key] = val

    # Mapping gem5 data columns to our columns
    df[CYCLE_COUNT] = int(data["board.processor.cores.core.numCycles"])
    df[IDLE_CYCLES] = int(data["board.processor.cores.core.idleCycles"])
    df[INSTR_COUNT] = int(data["board.processor.cores.core.commitStats0.numInsts"])
    df[INT_INSTR_COUNT] = int(data["board.processor.cores.core.commitStats0.numIntInsts"])
    df[FLOAT_INSTR_COUNT] = int(data["board.processor.cores.core.commitStats0.numFpInsts"])
    df[BRANCH_INSTR_COUNT] = int(data["board.processor.cores.core.executeStats0.numBranches"])
    df[BRANCH_MISPREDICTIONS] = int(data["board.processor.cores.core.commit.branchMispredicts"])
    df[LOADS] = int(data["board.processor.cores.core.commitStats0.numLoadInsts"])
    df[STORES] = int(data["board.processor.cores.core.commitStats0.numStoreInsts"])
    df[ROB_READS] = int(data["board.processor.cores.core.rob.reads"])
    df[ROB_WRITES] = int(data["board.processor.cores.core.rob.writes"])
    df[RENAME_READS] = int(data["board.processor.cores.core.rename.lookups"])
    df[RENAME_WRITES] = int(data["board.processor.cores.core.rename.renamedOperands"])
    df[FP_RENAME_READS] = int(data["board.processor.cores.core.rename.fpLookups"])
    df[INST_WINDOW_READS] = int(data["board.processor.cores.core.intInstQueueReads"]) + int(data["board.processor.cores.core.fpInstQueueReads"])
    df[INST_WINDOW_WRITES] = int(data["board.processor.cores.core.intInstQueueWrites"]) + int(data["board.processor.cores.core.fpInstQueueWrites"])
    df[INST_WINDOW_WAKEUP_ACCESSES] = int(data["board.processor.cores.core.intInstQueueWakeupAccesses"]) + int(data["board.processor.cores.core.fpInstQueueWakeupAccesses"])
    df[FP_INST_WINDOW_READS] = int(data["board.processor.cores.core.fpInstQueueReads"])
    df[FP_INST_WINDOW_WRITES] = int(data["board.processor.cores.core.fpInstQueueWrites"])
    df[FP_INST_WINDOW_WAKEUP_ACCESSES] = int(data["board.processor.cores.core.fpInstQueueWakeupAccesses"])

    df[INT_REGFILE_READS] = int(data["board.processor.cores.core.executeStats0.numIntRegReads"])
    df[INT_REGFILE_WRITES] = int(data["board.processor.cores.core.executeStats0.numIntRegWrites"])
    df[FLOAT_REGFILE_READS] = int(data["board.processor.cores.core.executeStats0.numFpRegReads"])
    df[FLOAT_REGFILE_WRITES] = int(data["board.processor.cores.core.executeStats0.numFpRegWrites"])
    df[FUNCTION_CALLS] = int(data["board.processor.cores.core.commit.functionCalls"])
    df[CONTEXT_SWITCHES] = int(data["board.processor.cores.core.commitStats0.committedControl::IsReturn"]) + int(data["board.processor.cores.core.commitStats0.committedControl::IsCall"])

    # TODO: is this wrong?
    df[MUL_ACCESS] = int(data["board.processor.cores.core.commitStats0.committedInstType::IntMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::FloatMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::SimdMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::SimdFloatMult"])

    df[FP_ACCESS] = int(data["board.processor.cores.core.fpAluAccesses"])
    df[IALU_ACCESS] = int(data["board.processor.cores.core.intAluAccesses"])

    df[BTB_READS] = int(data["board.processor.cores.core.branchPred.BTBLookups"])
    df[BTB_WRITES] = int(data["board.processor.cores.core.branchPred.BTBUpdates"])
    
    df[COMMITTED_INSTRUCTIONS] = self.total_instructions
    df[COMMITTED_INT_INSTRUCTIONS] = self.int_instructions
    df[COMMITTED_FLOAT_INSTRUCTIONS] = self.float_instructions
    df[FP_RENAME_WRITES] = self.fp_rename_reads // 2
    df[CDB_ALU_ACCESSES] = self.ialu_access
    df[CDB_FP_ACCESSES] = self.fp_access
    df[CDB_MUL_ACCESSES] = self.mul_access
    df[BUSY_CYCLES] = self.cycle_count - self.idle_cycles

    return pd.DataFrame(df)

def stats_df_estimate_missing_cols(df : pd.DataFrame):
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
    df[FP_INST_WINDOW_WAKEUP_ACCESSES] = df[FP_INST_WINDOW_READS] + df[FP_INST_WINDOW_WRITES]
    df[CDB_MUL_ACCESSES] = df[MUL_ACCESS]
    df[CDB_ALU_ACCESSES] = df[IALU_ACCESS]
    df[CDB_FP_ACCESSES] = df[FP_ACCESS]
    df[BTB_READS] = df[INSTR_COUNT]
    df[BTB_WRITES] = 0

    return df


def modify_xml(input_path: str, output_path: str, input_stats: dict, input_cfg, voltage_level: str) -> None:
    tree = ET.parse(input_path)

    VOLTAGE_LEVEL = MCPAT_VOLTAGE_MED

    if isinstance(voltage_level, str):
        if voltage_level == "high":
            VOLTAGE_LEVEL = MCPAT_VOLTAGE_HIGH
        elif voltage_level == "med":
            VOLTAGE_LEVEL = MCPAT_VOLTAGE_MED
        elif voltage_level == "low":
            VOLTAGE_LEVEL = MCPAT_VOLTAGE_LOW
        else:
            raise ValueError(f"Expected voltage level low/med/high, got {voltage_level}") 
    elif isinstance(voltage_level, int):
        VOLTAGE_LEVEL = [MCPAT_VOLTAGE_LOW, MCPAT_VOLTAGE_MED, MCPAT_VOLTAGE_HIGH][voltage_level]
    else:
        raise ValueError(f"Unrecognised voltage level input, got {voltage_level}")

    mcpat_config = input_cfg[MCPAT_CFG_MODULE_NAME]

    change_xml_property(tree, "system", "param", "temperature", str(mcpat_config[MCPAT_TEMP]))
    change_xml_property(tree, "system", "param", "core_tech_node", str(mcpat_config[MCPAT_NODE_SIZE]))
    change_xml_property(tree, "system", "param", "target_core_clockrate", str(mcpat_config[MCPAT_CLOCK_RATE]))
    # TODO: also run for low/high voltages
    change_xml_property(tree, "system", "param", "vdd", str(mcpat_config[MCPAT_VOLTAGE_MED]))
    
    change_xml_property(tree, "system", "stat", "total_cycles", str(input_stats[CYCLE_COUNT]))
    change_xml_property(tree, "system", "stat", "busy_cycles", str(input_stats[BUSY_CYCLES]))
    change_xml_property(tree, "system/system.core0", "param", "clock_rate", str(mcpat_config[MCPAT_CLOCK_RATE]))
    # TODO
    change_xml_property(tree, "system/system.core0", "stat", "total_instructions", str(input_stats[INSTR_COUNT]))
    change_xml_property(tree, "system/system.core0", "stat", "int_instructions", str(input_stats[INT_INSTR_COUNT]))
    change_xml_property(tree, "system/system.core0", "stat", "fp_instructions", str(input_stats[FLOAT_INSTR_COUNT]))
    change_xml_property(tree, "system/system.core0", "stat", "branch_instructions", str(input_stats[BRANCH_INSTR_COUNT]))
    change_xml_property(tree, "system/system.core0", "stat", "branch_mispredictions", str(input_stats[BRANCH_MISPREDICTIONS])) # TODO: assume some % of branches miss
    change_xml_property(tree, "system/system.core0", "stat", "load_instructions", str(input_stats[LOADS]))
    change_xml_property(tree, "system/system.core0", "stat", "store_instructions", str(input_stats[STORES]))
    change_xml_property(tree, "system/system.core0", "stat", "committed_instructions", str(input_stats[COMMITTED_INSTR]))
    change_xml_property(tree, "system/system.core0", "stat", "committed_int_instructions", str(input_stats[COMMITTED_INT_INSTR]))
    change_xml_property(tree, "system/system.core0", "stat", "committed_fp_instructions", str(input_stats[COMMITTED_FLOAT_INSTR]))
    change_xml_property(tree, "system/system.core0", "stat", "total_cycles", str(input_stats[CYCLE_COUNT]))
    change_xml_property(tree, "system/system.core0", "stat", "idle_cycles", str(input_stats[IDLE_CYCLES]))
    change_xml_property(tree, "system/system.core0", "stat", "busy_cycles", str(input_stats[BUSY_CYCLES]))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_reads", str(input_stats[ROB_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_writes", str(input_stats[ROB_WRITES]))
    change_xml_property(tree, "system/system.core0", "stat", "rename_reads", str(input_stats[RENAME_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "rename_writes", str(input_stats[RENAME_WRITES]))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_reads", str(input_stats[FP_RENAME_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_writes", str(input_stats[FP_RENAME_WRITES]))

    change_xml_property(tree, "system/system.core0", "stat", "inst_window_reads", str(input_stats[INST_WINDOW_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_writes", str(input_stats[INST_WINDOW_WRITES]))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_wakeup_accesses", str(input_stats[INST_WINDOW_WAKEUP_ACCESSES]))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_reads", str(input_stats[FP_INST_WINDOW_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_writes", str(input_stats[FP_INST_WINDOW_WRITES]))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_wakeup_accesses", str(input_stats[FP_INST_WINDOW_WAKEUP_ACCESSES]))

    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_reads", str(input_stats[INT_REGFILE_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_reads", str(input_stats[FLOAT_REGFILE_READS]))
    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_writes", str(input_stats[INT_REGFILE_WRITES]))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_writes", str(input_stats[FLOAT_REGFILE_WRITES]))
    change_xml_property(tree, "system/system.core0", "stat", "function_calls", str(input_stats[FUNCTION_CALLS]))
    change_xml_property(tree, "system/system.core0", "stat", "context_switches", str(input_stats[CONTEXT_SWITCHES]))
    change_xml_property(tree, "system/system.core0", "stat", "ialu_accesses", str(input_stats[IALU_ACCESS]))
    change_xml_property(tree, "system/system.core0", "stat", "fpu_accesses", str(input_stats[FP_ACCESS]))
    change_xml_property(tree, "system/system.core0", "stat", "mul_accesses", str(input_stats[MUL_ACCESS]))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_alu_accesses", str(input_stats[CDB_ALU_ACCESSES]))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_fpu_accesses", str(input_stats[CDB_FP_ACCESSES]))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_mul_accesses", str(input_stats[CDB_MUL_ACCESSES]))

    # TODO: missing stuff???
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_accesses", str(input_stats[INSTR_COUNT] // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_misses", str(4)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_accesses", str(input_stats[INSTR_COUNT] // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_misses", str(0)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_accesses", str(input_stats[INSTR_COUNT])) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_misses", str(0))

    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_accesses", str(input_stats[INSTR_COUNT] * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_misses", str(0))

    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "read_accesses", str(input_stats[BTB_READS]))
    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "write_accesses", str(input_stats[BTB_WRITES]))
    
    change_xml_property(tree, "system/system.L1Directory0", "param", "clockrate", str(mcpat_config[MCPAT_CLOCK_RATE]))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_accesses", str(input_stats[INSTR_COUNT] * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "conflicts", str(0))
    
    # TODO: all unsure
    change_xml_property(tree, "system/system.L2Directory0", "param", "clockrate", str(mcpat_config[MCPAT_CLOCK_RATE]))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L20", "param", "clockrate", str(mcpat_config[MCPAT_CLOCK_RATE]))
    change_xml_property(tree, "system/system.L20", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L30", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.NoC0", "stat", "total_accesses", str(input_stats[INSTR_COUNT] // 4)) # TODO: unsure
    change_xml_property(tree, "system/system.NoC0", "param", "clockrate", str(mcpat_config[MCPAT_CLOCK_RATE]))

    change_xml_property(tree, "system/system.mc", "stat", "memory_accesses", str(input_stats[INSTR_COUNT] // 10)) # TODO: unsure
    change_xml_property(tree, "system/system.mc", "stat", "memory_reads", str(input_stats[INSTR_COUNT] // 20))
    change_xml_property(tree, "system/system.mc", "stat", "memory_writes", str(input_stats[INSTR_COUNT] // 20))
    
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def load_arbitrary_stat_file(path: str, module_index: int=0, path_index: int=-1) -> pd.DataFrame:
    filename = os.path.basename(path)
    _, ext = os.path.splitext(filename)

    stats = None
    
    if filename == "PathBlocks.csv":
        all_stats = get_stats_df_subgraphs(path, module_index)

        if path_index == -1:
            raise NotImplementedError("Cannot sum over subgraphs, use MBB stats or select a specific subgraph")

        for stat_df in all_stats:
            if len(stat_df) == 1 and stat_df.iloc[0][PATH_INDEX] == path_index:
                stats = stat_df
                break

    elif filename == "MBB_stats.csv":
        all_stats = get_stats_df_mbbs(path, module_index)

        if path_index == -1:
            numerical_sums = all_stats.select_dtypes(include="number").sum()
            # Get first of all other columns
            other = all_stats.drop(columns=numerical_sums.index).iloc[0]
            stats = numerical_sums.combine_first(other)
            # Stats isn't strictly a dataframe at this point, but it works fine anyway 
            stats = stats.to_frame().T
        else:
            stats = all_stats[path_index]
    # Assuming gem5 output
    elif filename == "stats.txt":
        stats = get_stats_df_gem5_run(path)
    elif "_STD.csv" in filename:
        # TODO: warn on this path? or just check this path is valid
        stats = load_standard_stat_file(path)
    else:
        raise ValueError(f"Unknown stat file {filename}")

    if stats is None:
        raise RuntimeError("Failed to load stat file")

    # Convert from DF to series
    return stats.iloc[0]

def create_standard_stat_file(path: str, output_name: str, module_index: int=0, path_index: int=-1) -> None:
    stat_df = load_arbitrary_stat_file(path, module_index, path_index)
    stat_df.to_csv(f"{output_name}_STD.csv")

def load_standard_stat_file(path: str) -> dict:
    df = pd.read_csv(path, index_col=0)
    stats = df.iloc[:, 0].to_frame().T
    
    stats = stats.apply(pd.to_numeric, errors="ignore")
    # Cast all numeric columns to integers
    stats[stats.select_dtypes(include="number").columns] = (
        stats.select_dtypes(include="number").round().astype("Int64")
    )

    series = stats.iloc[0, :]

    return series.to_dict() 

