# TODO: import some xml stuff
import csv
import xml.etree.ElementTree as ET
import argparse
import re
import copy
# not really required, but useless for some csv processing
import pandas as pd

# TODO: note vdd is not present by default for AlphaXXXXX
# parameters that are not dependent on input file, but that we might want to change

DESIRED_CLOCK_RATE = 3400
DESIRED_CORE_TECH = 14 # nm
DESIRED_TEMP = 380 # k, increments of 10

def load_csv_to_dict(path: str):
    data = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data.append(row)

    return data

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

    xpath += f"/{param_or_stat}[@name='{name}']"

    element = tree.find(xpath)

    if element is None:
        raise ValueError(f"Tag [{xpath}] is not in XML")

    element.set("value", new_value) 

class RequiredStats:
    def __init__(self):
        self.cycle_count = 0
        self.idle_cycles = 0
        self.busy_cycles = 0
        self.total_instructions = 0
        self.int_instructions = 0
        self.float_instructions = 0
        self.branch_instructions = 0
        self.branch_mispredictions = 0
        self.load_instructions = 0
        self.store_instructions = 0
        self.committed_instructions = 0
        self.committed_float_instructions = 0
        self.committed_int_instructions = 0
        self.rob_reads = 0
        self.rob_writes = 0
        self.rename_reads = 0
        self.rename_writes = 0
        self.fp_rename_reads = 0
        self.fp_rename_writes = 0
        self.inst_window_wakeup_accesses = 0
        self.inst_window_writes = 0
        self.inst_window_reads = 0
        self.fp_inst_window_wakeup_accesses = 0
        self.fp_inst_window_writes = 0
        self.fp_inst_window_reads = 0
        self.int_regfile_reads = 0
        self.int_regfile_writes = 0
        self.float_regfile_reads = 0
        self.float_regfile_writes = 0
        self.function_calls = 0
        self.context_switches = 0
        self.mul_access = 0
        self.fp_access = 0
        self.ialu_access = 0
        self.cdb_mul_accesses = 0
        self.cdb_fp_accesses = 0
        self.cdb_alu_accesses = 0
        self.btb_reads = 0
        self.btb_writes = 0

    def _estimate_from_core_stats(self):
        """
        estimate the other stats from a couple important ones

        not necessarily good estimates
        """
        self.branch_mispredictions = 0
        self.idle_cycles = 0
        self.busy_cycles = self.cycle_count
        self.committed_instructions = self.total_instructions
        self.committed_int_instructions = self.int_instructions
        self.committed_float_instructions = self.float_instructions
        self.rename_writes = self.total_instructions
        self.rename_reads = self.total_instructions * 2
        self.rob_reads = self.total_instructions
        self.rob_writes = self.total_instructions
        self.fp_rename_reads = self.float_instructions * 2
        self.fp_rename_writes = self.float_instructions
        self.inst_window_reads = self.total_instructions
        self.inst_window_writes = self.total_instructions
        self.inst_window_wakeup_accesses = self.inst_window_reads + self.inst_window_writes
        self.fp_inst_window_reads = self.float_instructions
        self.fp_inst_window_writes = self.float_instructions
        self.fp_inst_window_wakeup_accesses = self.fp_inst_window_writes + self.fp_inst_window_reads
        self.cdb_mul_accesses = self.mul_access
        self.cdb_fp_accesses = self.fp_access
        self.cdb_alu_accesses = self.ialu_access
        self.btb_reads = self.total_instructions
        self.btb_writes = 0

    def load_csv_mbb_stats(self, csv_path, row_index=0):
        # TODO: use load_multipart_csv
        data = load_csv_to_dict(csv_path)

        mbb = data[row_index]

        self.cycle_count = int(float(mbb["cycles"]))
        self.total_instructions = int(float(mbb["instr_count"]))
        self.int_instructions = int(float(mbb["int_instr_count"]))
        self.float_instructions = int(float(mbb["float_instr_count"]))
        self.branch_instructions = int(float(mbb["branch_instr_count"]))
        self.load_instructions = int(float(mbb["loads"]))
        self.store_instructions = int(float(mbb["stores"]))
        self.int_regfile_reads = int(float(mbb["int_regfile_read"]))
        self.int_regfile_writes = int(float(mbb["int_regfile_write"]))
        self.float_regfile_reads = int(float(mbb["float_regfile_read"]))
        self.float_regfile_writes = int(float(mbb["float_regfile_write"]))
        self.function_calls = int(float(mbb["function_calls"]))
        self.context_switches = int(float(mbb["context_switches"]))
        self.mul_access = int(float(mbb["mul_access"]))
        self.fp_access = int(float(mbb["fp_access"]))
        self.ialu_access = int(float(mbb["ialu_access"]))

        self._estimate_from_core_stats()

    def load_sum_csv_mbb_stats(self, csv_path, module_index=0):
        data = load_multipart_csv(csv_path)
        
        module_data = data[module_index]
        
        module_df = pd.DataFrame(module_data)
        module_df = module_df.astype({
            "cycles": float,
            "instr_count": float,
            "int_instr_count": float,
            "float_instr_count": float,
            "branch_instr_count": float,
            "loads": float,
            "stores": float,
            "int_regfile_read": float,
            "int_regfile_write": float,
            "float_regfile_read": float,
            "float_regfile_write": float,
            "function_calls": float,
            "context_switches": float,
            "mul_access": float,
            "fp_access": float,
            "ialu_access": float
        })

        # Note we don't account for function frequency when calculating
        # execution frequency, we should be using GlobalFreq but
        # GlobalFreq has some high base multiplier we don't account for
        col_sums = module_df.sum()

        self.cycle_count = int(col_sums["cycles"])
        self.total_instructions = int(col_sums["instr_count"])
        self.int_instructions = int(col_sums["int_instr_count"])
        self.float_instructions = int(col_sums["float_instr_count"])
        self.branch_instructions = int(col_sums["branch_instr_count"])
        self.load_instructions = int(col_sums["loads"])
        self.store_instructions = int(col_sums["stores"])
        self.int_regfile_reads = int(col_sums["int_regfile_read"])
        self.int_regfile_writes = int(col_sums["int_regfile_write"])
        self.float_regfile_reads = int(col_sums["float_regfile_read"])
        self.float_regfile_writes = int(col_sums["float_regfile_write"])
        self.function_calls = int(col_sums["function_calls"])
        self.context_switches = int(col_sums["context_switches"])
        self.mul_access = int(col_sums["mul_access"])
        self.fp_access = int(col_sums["fp_access"])
        self.ialu_access = int(col_sums["ialu_access"])
        
        self._estimate_from_core_stats()

    def load_csv_crit_path(self, csv_path, module_index=0, crit_path_index=0):
        # NOTE: function is a bit useless, needing this arbitrary module and crit path index
        data = load_multipart_csv(csv_path)

        crit_paths = data[module_index]
        crit_path = crit_paths[crit_path_index]

        self.cycle_count = int(float(crit_path["cycles"]))
        self.total_instructions = int(float(crit_path["instrs"]))
        self.int_instructions = int(float(crit_path["int_instr"]))
        self.float_instructions = int(float(crit_path["float_instr"]))
        self.branch_instructions = int(float(crit_path["branch_instr"]))
        self.load_instructions = int(float(crit_path["loads"]))
        self.store_instructions = int(float(crit_path["stores"]))
        self.int_regfile_reads = int(float(crit_path["int_regfile_read"]))
        self.int_regfile_writes = int(float(crit_path["int_regfile_write"]))
        self.float_regfile_reads = int(float(crit_path["float_regfile_read"]))
        self.float_regfile_writes = int(float(crit_path["float_regfile_write"]))
        self.function_calls = int(float(crit_path["func_calls"]))
        self.context_switches = int(float(crit_path["context_switches"]))
        self.mul_access = int(float(crit_path["mul_access"]))
        self.fp_access = int(float(crit_path["fp_access"]))
        self.ialu_access = int(float(crit_path["int_alu_access"]))

        self._estimate_from_core_stats()

    def load_gem5_stats(self, input_path):
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

        # mapping from data to the variables
        # TODO: a lot of these unsure of
        self.cycle_count = int(data["board.processor.cores.core.numCycles"]) 
        self.idle_cycles = int(data["board.processor.cores.core.idleCycles"])
        self.busy_cycles = self.cycle_count - self.idle_cycles
        self.total_instructions = int(data["board.processor.cores.core.commitStats0.numInsts"])
        self.int_instructions = int(data["board.processor.cores.core.commitStats0.numIntInsts"])
        self.float_instructions = int(data["board.processor.cores.core.commitStats0.numFpInsts"])
        self.branch_instructions = int(data["board.processor.cores.core.executeStats0.numBranches"])
        self.branch_mispredictions = int(data["board.processor.cores.core.commit.branchMispredicts"])
        self.load_instructions = int(data["board.processor.cores.core.commitStats0.numLoadInsts"])
        self.store_instructions = int(data["board.processor.cores.core.commitStats0.numStoreInsts"])
        self.committed_instructions = self.total_instructions
        self.committed_int_instructions = self.int_instructions
        self.committed_float_instructions = self.float_instructions
        self.rob_reads = int(data["board.processor.cores.core.rob.reads"])
        self.rob_writes = int(data["board.processor.cores.core.rob.writes"])
        self.rename_reads = int(data["board.processor.cores.core.rename.lookups"])
        self.rename_writes = int(data["board.processor.cores.core.rename.renamedOperands"])
        self.fp_rename_reads = int(data["board.processor.cores.core.rename.fpLookups"])
        self.fp_rename_writes = self.fp_rename_reads // 2 #  NOTE: this is an estimation!
        self.inst_window_reads = int(data["board.processor.cores.core.intInstQueueReads"]) + int(data["board.processor.cores.core.fpInstQueueReads"])
        self.inst_window_writes = int(data["board.processor.cores.core.intInstQueueWrites"]) + int(data["board.processor.cores.core.fpInstQueueWrites"])
        self.inst_window_wakeup_accesses = int(data["board.processor.cores.core.intInstQueueWakeupAccesses"]) + int(data["board.processor.cores.core.fpInstQueueWakeupAccesses"])
        self.fp_inst_window_reads = int(data["board.processor.cores.core.fpInstQueueReads"])
        self.fp_inst_window_writes = int(data["board.processor.cores.core.fpInstQueueWrites"])
        self.fp_inst_window_wakeup_accesses = int(data["board.processor.cores.core.fpInstQueueWakeupAccesses"])

        self.int_regfile_reads = int(data["board.processor.cores.core.executeStats0.numIntRegReads"])
        self.int_regfile_writes = int(data["board.processor.cores.core.executeStats0.numIntRegWrites"])
        self.float_regfile_reads = int(data["board.processor.cores.core.executeStats0.numFpRegReads"])
        self.float_regfile_writes = int(data["board.processor.cores.core.executeStats0.numFpRegWrites"])
        self.function_calls = int(data["board.processor.cores.core.commit.functionCalls"])
        self.context_switches = int(data["board.processor.cores.core.commitStats0.committedControl::IsReturn"]) + int(data["board.processor.cores.core.commitStats0.committedControl::IsCall"])

        # TODO: is this wrong?
        self.mul_access = int(data["board.processor.cores.core.commitStats0.committedInstType::IntMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::FloatMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::SimdMult"]) + int(data["board.processor.cores.core.commitStats0.committedInstType::SimdFloatMult"])

        self.fp_access = int(data["board.processor.cores.core.fpAluAccesses"])
        self.ialu_access = int(data["board.processor.cores.core.intAluAccesses"])

        self.cdb_alu_accesses = self.ialu_access
        self.cdb_fp_accesses = self.fp_access
        self.cdb_mul_accesses = self.mul_access

        self.btb_reads = int(data["board.processor.cores.core.branchPred.BTBLookups"])
        self.btb_writes = int(data["board.processor.cores.core.branchPred.BTBUpdates"])

def modify_xml(input_path: str, output_path: str, input_stats: str, index: int) -> None:
    tree = ET.parse(input_path)

    stats = RequiredStats()
    
    if input_stats.endswith(".csv"):
        if input_stats.endswith("CritPath.csv", index):
            stats.load_csv_crit_path(input_stats)
        elif input_stats.endswith("MBB_stats.csv", index):
            stats.load_csv_mbb_stats(input_stats)
        else:
            print("[WARN] unsure if input file is CritPath or MBB, assuming MBB")
            stats.load_csv_mbb_stats(input_stats)
    else:
        stats.load_gem5_stats(input_stats)

    change_xml_property(tree, "system", "param", "temperature", str(DESIRED_TEMP))
    change_xml_property(tree, "system", "param", "core_tech_node", str(DESIRED_CORE_TECH))
    change_xml_property(tree, "system", "param", "target_core_clockrate", str(DESIRED_CLOCK_RATE))
    
    change_xml_property(tree, "system", "stat", "total_cycles", str(stats.cycle_count))
    change_xml_property(tree, "system", "stat", "busy_cycles", str(stats.cycle_count))
    change_xml_property(tree, "system/system.core0", "param", "clock_rate", str(DESIRED_CLOCK_RATE))
    # TODO
    change_xml_property(tree, "system/system.core0", "stat", "total_instructions", str(stats.total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "int_instructions", str(stats.int_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "fp_instructions", str(stats.float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "branch_instructions", str(stats.branch_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "branch_mispredictions", str(stats.branch_mispredictions)) # TODO: assume some % of branches miss
    change_xml_property(tree, "system/system.core0", "stat", "load_instructions", str(stats.load_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "store_instructions", str(stats.store_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_instructions", str(stats.committed_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_int_instructions", str(stats.committed_int_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_fp_instructions", str(stats.committed_float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "total_cycles", str(stats.cycle_count))
    change_xml_property(tree, "system/system.core0", "stat", "idle_cycles", str(stats.idle_cycles))
    change_xml_property(tree, "system/system.core0", "stat", "busy_cycles", str(stats.busy_cycles))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_reads", str(stats.rob_reads))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_writes", str(stats.rob_writes))
    change_xml_property(tree, "system/system.core0", "stat", "rename_reads", str(stats.rename_reads))
    change_xml_property(tree, "system/system.core0", "stat", "rename_writes", str(stats.rename_writes))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_reads", str(stats.fp_rename_reads))
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_writes", str(stats.fp_rename_writes))

    change_xml_property(tree, "system/system.core0", "stat", "inst_window_reads", str(stats.inst_window_reads))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_writes", str(stats.inst_window_writes))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_wakeup_accesses", str(stats.inst_window_wakeup_accesses))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_reads", str(stats.fp_inst_window_reads))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_writes", str(stats.fp_inst_window_writes))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_wakeup_accesses", str(stats.fp_inst_window_wakeup_accesses))

    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_reads", str(stats.int_regfile_reads))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_reads", str(stats.float_regfile_reads))
    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_writes", str(stats.int_regfile_writes))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_writes", str(stats.float_regfile_writes))
    change_xml_property(tree, "system/system.core0", "stat", "function_calls", str(stats.function_calls))
    change_xml_property(tree, "system/system.core0", "stat", "context_switches", str(stats.context_switches))
    change_xml_property(tree, "system/system.core0", "stat", "ialu_accesses", str(stats.ialu_access))
    change_xml_property(tree, "system/system.core0", "stat", "fpu_accesses", str(stats.fp_access))
    change_xml_property(tree, "system/system.core0", "stat", "mul_accesses", str(stats.mul_access))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_alu_accesses", str(stats.cdb_alu_accesses))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_fpu_accesses", str(stats.cdb_fp_accesses))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_mul_accesses", str(stats.cdb_mul_accesses))

    # TODO: missing stuff???
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_accesses", str(stats.total_instructions // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_misses", str(4)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_accesses", str(stats.total_instructions // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_misses", str(0)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_accesses", str(stats.total_instructions)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_misses", str(0))

    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_accesses", str(stats.total_instructions * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_misses", str(0))

    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "read_accesses", str(stats.btb_reads))
    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "write_accesses", str(stats.btb_writes))
    
    change_xml_property(tree, "system/system.L1Directory0", "param", "clockrate", str(DESIRED_CLOCK_RATE))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_accesses", str(stats.total_instructions * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "conflicts", str(0))
    
    # TODO: all unsure
    change_xml_property(tree, "system/system.L2Directory0", "param", "clockrate", str(DESIRED_CLOCK_RATE))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L20", "param", "clockrate", str(DESIRED_CLOCK_RATE))
    change_xml_property(tree, "system/system.L20", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L30", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.NoC0", "stat", "total_accesses", str(stats.total_instructions // 4)) # TODO: unsure
    change_xml_property(tree, "system/system.NoC0", "param", "clockrate", str(DESIRED_CLOCK_RATE))

    change_xml_property(tree, "system/system.mc", "stat", "memory_accesses", str(stats.total_instructions // 10)) # TODO: unsure
    change_xml_property(tree, "system/system.mc", "stat", "memory_reads", str(stats.total_instructions // 20))
    change_xml_property(tree, "system/system.mc", "stat", "memory_writes", str(stats.total_instructions // 20))
    
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def main():
    parser = argparse.ArgumentParser(description="Create xml file based on Alpha given some data.")
    parser.add_argument("input_stats", help="Path to csv file CritPath.csv/MBB_stats.csv/gem5 output file stats.txt")
    parser.add_argument("input_xml", help="Path to input XML (expecting Alpha21364.xml) file")
    parser.add_argument("output_xml", help="Name of output XML file")
    parser.add_argument("index", help="(Optional) Which basic block or critical path to read", type=int, default=0)
    args = parser.parse_args()

    modify_xml(args.input_xml, args.output_xml, args.input_stats, args.index)

if __name__ == "__main__":
    main()


