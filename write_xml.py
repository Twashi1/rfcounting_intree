# TODO: import some xml stuff
import csv
import xml.etree.ElementTree as ET
import argparse

# TODO: note vdd is not present by default for AlphaXXXXX
# parameters that are not dependent on input file, but that we might want to change
# TODO: just move to how we set the other things
default_params = {
    "./component[@id='system'][@name='system']/param[@name='temperature']": "380",
    "./component[@id='system'][@name='system']/param[@name='core_tech_node']": "14", # NOTE: changed from 180
    "./component[@id='system'][@name='system']/param[@name='target_core_clockrate']": "3400", # NOTE: changed from 1200
}

# TODO: just wrote this randomly, but write in some other things in this format, and set later instead of default_params
DESIRED_CLOCK_RATE = 3400

def load_csv_to_dict(path: str):
    data = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data.append(row)

    return data

def change_xml_property(tree, component_path: str, param_or_stat: str, name: str, new_value: str):
    xpath = "."

    for id in component_path.split("/"):
        xpath += f"/component[@id='{id}']"

    xpath += f"/{param_or_stat}[@name='{name}']"

    element = tree.find(xpath)

    if element is None:
        raise ValueError(f"Tag [{xpath}] is not in XML")

    element.set("value", new_value) 

def modify_xml(input_path: str, output_path: str, input_csv: str):
    tree = ET.parse(input_path)

    # set default
    for path, value in default_params.items():
        param = tree.find(path)

        if param is None:
            raise ValueError(f"Attempted to grab path {path} but failed!")

        param.set("value", value)

    # TODO: read from CritPath.csv and set other values
    records = load_csv_to_dict(input_csv)
    first_crit_path = records[0]

    cycle_count = int(float(first_crit_path["cycles"]))
    total_instructions = int(float(first_crit_path["instrs"]))
    int_instructions  = int(float(first_crit_path["int_instr"]))
    float_instructions = int(float(first_crit_path["float_instr"]))
    branch_instructions = int(float(first_crit_path["branch_instr"]))
    load_instructions = int(float(first_crit_path["loads"]))
    store_instructions = int(float(first_crit_path["stores"]))
    branch_instructions = int(float(first_crit_path["branch_instr"]))
    int_regfile_reads = int(float(first_crit_path["int_regfile_read"]))
    int_regfile_writes = int(float(first_crit_path["int_regfile_write"]))
    float_regfile_reads = int(float(first_crit_path["float_regfile_read"]))
    float_regfile_writes = int(float(first_crit_path["float_regfile_write"]))
    function_calls = int(float(first_crit_path["func_calls"]))
    context_switches = int(float(first_crit_path["context_switches"]))
    mul_access = int(float(first_crit_path["mul_access"]))
    fp_access = int(float(first_crit_path["fp_access"]))
    ialu_access = int(float(first_crit_path["int_alu_access"]))
    
    change_xml_property(tree, "system", "stat", "total_cycles", str(cycle_count))
    change_xml_property(tree, "system", "stat", "busy_cycles", str(cycle_count))
    change_xml_property(tree, "system/system.core0", "param", "clock_rate", str(3400))
    # TODO
    change_xml_property(tree, "system/system.core0", "stat", "total_instructions", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "int_instructions", str(int_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "fp_instructions", str(float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "branch_instructions", str(branch_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "branch_mispredictions", str(0)) # TODO: assume some % of branches miss
    change_xml_property(tree, "system/system.core0", "stat", "load_instructions", str(load_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "store_instructions", str(store_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_instructions", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_int_instructions", str(int_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "committed_fp_instructions", str(float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "total_cycles", str(cycle_count))
    change_xml_property(tree, "system/system.core0", "stat", "busy_cycles", str(cycle_count))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_reads", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "ROB_writes", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "rename_reads", str(total_instructions * 2))
    change_xml_property(tree, "system/system.core0", "stat", "rename_writes", str(total_instructions))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_reads", str(float_instructions * 2))
    change_xml_property(tree, "system/system.core0", "stat", "fp_rename_writes", str(float_instructions))

    change_xml_property(tree, "system/system.core0", "stat", "inst_window_reads", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_writes", str(total_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "inst_window_wakeup_accesses", str(total_instructions * 2))
    # TODO: unsure about below
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_reads", str(float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_writes", str(float_instructions))
    change_xml_property(tree, "system/system.core0", "stat", "fp_inst_window_wakeup_accesses", str(float_instructions * 2))

    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_reads", str(int_regfile_reads))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_reads", str(float_regfile_reads))
    change_xml_property(tree, "system/system.core0", "stat", "int_regfile_writes", str(int_regfile_writes))
    change_xml_property(tree, "system/system.core0", "stat", "float_regfile_writes", str(float_regfile_writes))
    change_xml_property(tree, "system/system.core0", "stat", "function_calls", str(function_calls))
    change_xml_property(tree, "system/system.core0", "stat", "context_switches", str(context_switches))
    change_xml_property(tree, "system/system.core0", "stat", "ialu_accesses", str(ialu_access))
    change_xml_property(tree, "system/system.core0", "stat", "fpu_accesses", str(fp_access))
    change_xml_property(tree, "system/system.core0", "stat", "mul_accesses", str(mul_access))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_alu_accesses", str(ialu_access)) # TODO: unsure
    change_xml_property(tree, "system/system.core0", "stat", "cdb_fpu_accesses", str(fp_access))
    change_xml_property(tree, "system/system.core0", "stat", "cdb_mul_accesses", str(mul_access))

    # TODO: missing stuff???
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_accesses", str(total_instructions // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "total_misses", str(4)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.itlb", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_accesses", str(total_instructions // 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "read_misses", str(0)) # TODO: some small default
    change_xml_property(tree, "system/system.core0/system.core0.icache", "stat", "conflicts", str(0))
    
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_accesses", str(total_instructions)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dtlb", "stat", "total_misses", str(3400))

    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_accesses", str(total_instructions * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.core0/system.core0.dcache", "stat", "write_misses", str(0))

    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "read_accesses", str(total_instructions)) # TODO: unsure
    change_xml_property(tree, "system/system.core0/system.core0.BTB", "stat", "write_accesses", str(0))
    
    change_xml_property(tree, "system/system.L1Directory0", "param", "clockrate", str(3400))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_accesses", str(total_instructions * 2)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_accesses", str(0)) # TODO: unsure
    change_xml_property(tree, "system/system.L1Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "write_misses", str(0))
    change_xml_property(tree, "system/system.L1Directory0", "stat", "conflicts", str(0))
    
    # TODO: all unsure
    change_xml_property(tree, "system/system.L2Directory0", "param", "clockrate", str(3400))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L2Directory0", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L20", "param", "clockrate", str(3400))
    change_xml_property(tree, "system/system.L20", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L20", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.L30", "stat", "read_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_accesses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "read_misses", str(0))
    change_xml_property(tree, "system/system.L30", "stat", "write_misses", str(0))
    
    change_xml_property(tree, "system/system.NoC0", "stat", "total_accesses", str(total_instructions // 4)) # TODO: unsure
    change_xml_property(tree, "system/system.NoC0", "param", "clockrate", str(3400))

    change_xml_property(tree, "system/system.mc", "stat", "memory_accesses", str(total_instructions // 10)) # TODO: unsure
    change_xml_property(tree, "system/system.mc", "stat", "memory_reads", str(total_instructions // 20))
    change_xml_property(tree, "system/system.mc", "stat", "memory_writes", str(total_instructions // 20))
    
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def main():
    parser = argparse.ArgumentParser(description="Create xml file based on Alpha given some data.")
    parser.add_argument("csv_file", help="Path to csv file")
    parser.add_argument("input_xml", help="Path to input Alpha??? file")
    parser.add_argument("output_xml", help="Name of output XML file")
    args = parser.parse_args()

    modify_xml(args.input_xml, args.output_xml, args.csv_file)

if __name__ == "__main__":
    main()


