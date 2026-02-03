# TODO: read in some folder of mcpat .txt inputs, and convert to ptrace for HotSpot
# TODO: also requires the HotSpot floorplan
import utils
import argparse
import re
import pandas as pd
import os
import shutil
import subprocess
from collections import defaultdict, deque

AGGREGATE_METHOD_WORST_CASE = "hottest"
AGGREGATE_METHOD_MOST_LIKELY = "likely"
AGGREGATE_METHOD_WEIGHTED_AVG = "weighted"

HOTSPOT_FLOORPLAN = {
    "L2_left",
    "L2",
    "L2_right",
    "Icache",
    "Dcache",
    "Bpred_0",
    "Bpred_1",
    "Bpred_2",
    "DTB_0",
    "DTB_1",
    "DTB_2",
    "FPAdd_0",
    "FPAdd_1",
    "FPReg_0",
    "FPReg_1",
    "FPReg_2",
    "FPReg_3",
    "FPMul_0",
    "FPMul_1",
    "FPMap_0",
    "FPMap_1",
    "IntMap",
    "IntQ",
    "IntReg_0",
    "IntReg_1",
    "IntExec",
    "FPQ",
    "LdStQ",
    "ITB_0",
    "ITB_1",
}


def celsius_to_kelvin(celsius: float):
    return celsius + 273.15


# TODO: "name" is actually name prefix
def mcpat_get_unit_stats(name: str, text: str) -> dict:
    # Note sometimes a colon, sometimes not
    pattern = rf"\s*{re.escape(name)}.*\n(\s+.+\s=\s.+\n)+"

    m = re.search(pattern, text, re.MULTILINE)

    if not m:
        return None

    block = m.group(1)
    fields = {}

    # print(f"{name=} {block=}")

    for key in [
        "Area",
        "Peak Dynamic",
        "Subthreshold Leakage",
        "Gate Leakage",
        "Runtime Dynamic",
    ]:
        fm = re.search(rf"^\s+{key}\s*=\s*(\S+)\s.+$", block)
        if fm:
            # print(f"{fm.group(1)=}")
            fields[key] = float(fm.group(1))

    # print(f"{fields=}")

    return fields


def mcpat_to_dict(mcpat_out_file: str) -> dict:
    text = ""

    with open(mcpat_out_file, "r") as f:
        text = f.read()

    unit_stats = {}

    for unit in [
        "L2",
        "Instruction Cache",
        "Data Cache",
        "Branch Predictor",
        "Dtlb",
        "Floating Point Unit",
        "Floating Point RF",
        "Renaming Unit",
        "Int Front End RAT",
        "FP Front End RAT",
        "Int Retire RAT",
        "FP Retire RAT",
        "FP Free List",
        "Free List",  # assuming this is only integer free list?
        "Instruction Scheduler",
        "Instruction Window",  # assuming this is only integer instruction window?
        "FP Instruction Window",
        "Integer RF",
        "Integer ALU",
        "Load Store Unit",
        "LoadQ",
        "StoreQ",
        "Itlb",
    ]:
        stats = mcpat_get_unit_stats(unit, text)
        unit_stats[unit] = stats

    # TODO: make this a dedicated function, deal with multiple sum behaviour more explicitly
    unit_power = lambda key: (
        unit_stats[key]["Runtime Dynamic"]
        if isinstance(key, str)
        else sum(unit_stats[i]["Runtime Dynamic"] for i in key)
    )

    # Now map unit stats to hotspot
    # Assuming Alpha EV6 floorplan
    # TODO: Alpha EV6 seems to be different to the Alpha21364.xml we use as our baseline?
    # Note we don't have a direct mapping, so we take some averages
    # TODO: mathematical correctness of averaging? shouldn't we consider it to just be one component with less heat spread?
    # TODO: unsure about which McPAT components to use
    hotspot_mapping = {
        "L2_left": unit_power("L2") / 3.0,
        "L2": unit_power("L2") / 3.0,
        "L2_right": unit_power("L2") / 3.0,
        "Icache": unit_power("Instruction Cache"),
        "Dcache": unit_power("Data Cache"),
        "Bpred_0": unit_power("Branch Predictor") / 3.0,
        "Bpred_1": unit_power("Branch Predictor") / 3.0,
        "Bpred_2": unit_power("Branch Predictor") / 3.0,
        "DTB_0": unit_power("FP Front End RAT") / 3.0,
        "DTB_1": unit_power("Dtlb") / 3.0,
        "DTB_2": unit_power("Dtlb") / 3.0,
        # NOTE: averaging these ones over the Add and Mul units
        "FPAdd_0": unit_power("Floating Point Unit") / 4.0,
        "FPAdd_1": unit_power("Floating Point Unit") / 4.0,
        "FPReg_0": unit_power("Floating Point RF") / 4.0,
        "FPReg_1": unit_power("Floating Point RF") / 4.0,
        "FPReg_2": unit_power("Floating Point RF") / 4.0,
        "FPReg_3": unit_power("Floating Point RF") / 4.0,
        "FPMul_0": unit_power("Floating Point Unit") / 4.0,
        "FPMul_1": unit_power("Floating Point Unit") / 4.0,
        "FPMap_0": unit_power(("FP Front End RAT", "FP Retire RAT", "FP Free List"))
        / 2.0,
        "FPMap_1": unit_power(("FP Front End RAT", "FP Retire RAT", "FP Free List"))
        / 2.0,
        "IntMap": unit_power(("Int Front End RAT", "Int Retire RAT", "Free List")),
        "IntQ": unit_power("Instruction Window"),
        "IntReg_0": unit_power("Integer RF") / 2.0,
        "IntReg_1": unit_power("Integer RF") / 2.0,
        "IntExec": unit_power("Integer ALU"),
        "FPQ": unit_power("FP Instruction Window"),
        "LdStQ": unit_power(("LoadQ", "StoreQ")),
        "ITB_0": unit_power("Itlb") / 2.0,
        "ITB_1": unit_power("Itlb") / 2.0,
    }

    return hotspot_mapping


def load_folder_mcpat(folder_path: str, file_prefix: str) -> pd.DataFrame:
    folder_path = folder_path.rstrip(os.sep)
    last_directory_name = os.path.basename(folder_path)

    if file_prefix == "":
        file_prefix = last_directory_name

    if file_prefix == "":
        raise ValueError(f"Bad folder path? File prefix was empty. {folder_path=}")

    pattern = re.compile(
        rf"{re.escape(file_prefix)}_idx(\d{{4}})_((high|med|low))\.txt$"
    )

    all_data = defaultdict(list)

    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)

        if not os.path.isfile(file_path):
            continue

        hotspot_mappings = mcpat_to_dict(file_path)

        match = pattern.match(f)

        if not match:
            raise ValueError(f"Unrecognised file name format: {f}, in path {file_path}")

        path_number = int(match.group(1))
        voltage_level = match.group(2)

        unit_stats = mcpat_to_dict(file_path)

        all_data["voltage"].append(voltage_level)

        # TODO: we don't know if these are paths or blocks, so we have to add both
        all_data["path_index"].append(path_number)
        all_data["block_id"].append(path_number)

        for key, power in unit_stats.items():
            all_data[key].append(power)

    return pd.DataFrame(all_data)


def get_hotspot_temp(
    hotspot_ptrace: dict,
    execution_time: float,
    config_file: str,
    initial_heat: dict,
) -> dict:
    # TODO: construct input from hotspot_ptrace (dataframe series, or turn to dict?)
    # TODO: assume ttrace is a file path to use as input, initial_temp is numerical (kelvin)
    # TODO: call hotspot binary with the inputs
    # TODO: load outputs as ttrace output path?
    with open("./hotspot_files/gcc.ptrace", "w") as f:
        # Write in keys
        i = 0
        for key in hotspot_ptrace.keys():
            if key not in HOTSPOT_FLOORPLAN:
                continue

            if i > 0:
                f.write("\t")

            i += 1

            f.write(key)

        f.write("\n")

        # Write in values
        # TODO: only write the relevant columns
        i = 0
        for key in hotspot_ptrace.keys():
            if key not in HOTSPOT_FLOORPLAN:
                continue

            if i > 0:
                f.write("\t")

            i += 1

            value = hotspot_ptrace[key]
            value = float(value)

            f.write(f"{value:.6f}")

        f.write("\n")

    if initial_heat is not None:
        # Create input file
        # TODO: just write directly, this is some stupidity from old code
        write_heatmap_to_init("./hotspot_files/block_in.init", initial_heat)
        shutil.copyfile("./hotspot_files/block_in.init", "./hotspot_files/gcc.init")

    # Replace line in example.config
    lines = []

    with open(config_file, "r") as f:
        lines = f.readlines()

    pattern = re.compile(r"^\s*sampling_intvl\s+[\d.eE+-]+")

    # TODO: check whitespaces are correct
    # TODO: verify it correctly replaces the text
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"\t-sampling_intvl\t\t{execution_time}\n"
            break

    with open(config_file, "w") as f:
        f.writelines(lines)

    # Run hotspot
    should_use_initial = "true" if initial_heat is not None else "false"
    subprocess.run(["./run_hotspot.sh", should_use_initial], check=True)

    # Read the output file and save
    heatmap_output = read_heatmap_output("./hotspot_files/outputs/gcc.ttrace")

    return heatmap_output


def ordered_nodes(scc_topo: list, comp_to_nodes: dict, adj: pd.DataFrame):
    # print(f"{scc_topo=}\n{comp_to_nodes=}\n{adj=}")

    topo_sorted = []

    for comp in scc_topo:
        nodes = comp_to_nodes[comp]

        # Kahn's topo sort, roughly same as implementation used in RegisterAccessPreRAPass.cpp
        indegree = defaultdict(int)
        node_set = set(nodes)

        for u in nodes:
            for v in adj[u]:
                if v not in node_set:
                    continue

                # add dependency
                indegree[v] += 1

        q = deque([n for n in nodes if indegree[n] == 0])
        seen = set(q)

        while q:
            u = q.popleft()

            topo_sorted.append(u)

            for v in adj[u]:
                if v not in node_set:
                    continue

                # fulfill dependency
                indegree[v] -= 1

                if indegree[v] == 0 and v not in seen:
                    seen.add(v)
                    q.append(v)

        # TODO: set intersection of node_set and seen is probably faster?
        # Not perfect with intra-SCC connections
        # TODO: not necessarily in order of index/id value, so should sort?
        for n in nodes:
            if n not in seen:
                topo_sorted.append(n)
                seen.add(n)

    return topo_sorted


def aggregate_weighted_average(heats, branch_prob):
    new_heat = defaultdict(float)

    for i, heat in enumerate(heats):
        for key, value in heat.items():
            new_heat[key] += value * branch_prob

    return new_heat


def aggregate_most_likely(heats, branch_prob):
    # pythonic or ugly?
    max_index, max_heat = max(enumerate(heats), key=lambda p: branch_prob[p[0]])

    return max_heat


def aggregate_worst_case(heats):
    new_heat = defaultdict(float)

    # Maximum heat for every heat
    for i, heat in enumerate(heats):
        for key, value in heat.items():
            new_heat[key] = max(new_heat[key], value)

    return new_heat


def aggregate_heat_data(
    heat_data: dict,
    nodes: list,
    target: int,
    block_additional: pd.DataFrame,
    full_cfg: pd.DataFrame,
    method: str,
):
    # Filter all nodes with no heat data
    filtered = [node for node in nodes if node in heat_data]
    # Get heat data of filtered nodes
    heats = [heat_data[node] for node in filtered]
    # Compute branch probabilities for each edge (used in aggregates)
    branch_prob = []

    if len(filtered) == 0:
        return None

    for start_block_id in filtered:
        prob = df.loc[
            (df["start_block_id"] == start_block_id) & (df["exit_block_id"] == target),
            "branch_prob",
        ].iloc[0]

        branch_prob.append(prob)

    if abs(sum(branch_prob) - 1.0) > 0.05:
        print("[WARN] Branch probabilities summed to not close to 1! Should normalise?")

        # Normalisation
        total = sum(branch_prob)
        branch_prob = [i / total for i in branch_prob]

    selected_heat = None

    if method == AGGREGATE_METHOD_MOST_LIKELY:
        selected_heat = aggregate_most_likely(heats, branch_prob)
    elif method == AGGREGATE_METHOD_WEIGHTED_AVG:
        selected_heat = aggregate_weighted_average(heats, branch_prob)
    elif method == AGGREGATE_METHOD_WORST_CASE:
        selected_heat = aggregate_worst_case(heats)
    else:
        raise ValueError(f"Invalid aggregate method, check name: {method}")

    return selected_heat


def write_heatmap_to_init(file_path: str, heat: dict) -> None:
    # Write values for each key
    # then write
    # we don't have any real values for the ones below? so we just guess them
    # TODO: there is almost surely a way to get the correct output such that we can give more valid inputs for these
    # - iface_
    # - hsp_
    # - hsink_
    # - inode_0 through inode_11 (unsure of correct values)
    with open(file_path, "w") as f:
        for key, value in heat.items():
            f.write(f"{key} {value:.2f}\n")

        for key, value in heat.items():
            f.write(f"iface_{key} {value:.2f}\n")

        for key, value in heat.items():
            f.write(f"hsp_{key} {value:.2f}\n")

        for key, value in heat.items():
            f.write(f"hsink_{key} {value:.2f}\n")

        # TODO: we have no clue of correct value, so we assume temperature of 75
        # (just assuming something pretty warm)
        for i in range(12):
            f.write(f"inode_{i} {celsius_to_kelvin(75.00):.2f}")


def read_heatmap_output(file_path: str) -> dict:
    # Reading a ttrace file (separated by space, regex to be safe)
    df = pd.read_csv(file_path, sep=r"\s+")
    # All columns numerical, so should work
    df = df.astype(float)

    if len(df) > 1:
        print("[WARN] DF had more than one row, we just take the first")

    return df.iloc[0].to_dict()


def calculate_all_heat(
    mcpat_df: pd.DataFrame,
    approx_sorted_nodes: list,
    global_adj: dict,
    additional_block: pd.DataFrame,
    cfg: pd.DataFrame,
    dag: pd.DataFrame,
    aggregate_method: str,
    clock_rate: float,
    config_file: str,
):
    # Clock rate given in MHz, need GHz
    clock_rate = float(clock_rate)
    clock_rate *= 1.0e6

    # Add execution time column to the mcpat df
    merged = mcpat_df.merge(additional_block, on="block_id")

    # TODO: implementation
    # - calculate the backwards mapping from a node to all its parents
    # - we are given the order to compute nodes in
    # - we iterate this order
    # - for each node, look at all its parents, and grab the heat data (stored here too)
    # - get hotspot_temp given the heat data, execution time, etc.
    # - store this heat data

    # Store all heat data for each node
    heat_data = dict()

    # Create backwards mapping from node to parents
    parents = defaultdict(list)

    for start in cfg.keys():
        for end in cfg[start]:
            parents[end].append(start)

    for node in approx_sorted_nodes:
        # Get all parents
        prevs = parents[node]
        # Get heat data of all parents
        parent_heats = {
            parent: heat_data[parent] for parent in heat_data if parent in heat_data
        }

        # Get the new heat
        new_heat = aggregate_heat_data(
            parent_heats, prevs, node, additional_block, cfg, aggregate_method
        )

        # TODO: output new_heat as input for hotspot
        # We were not able to get any priors to estimate heat from
        if new_heat is None or len(new_heat) == 0:
            new_heat = None

        # Node is our block id, get execution time data for it, and the mcpat row data
        print(f"Testing for block id: {node=}")
        mcpat_ptrace = merged.loc[merged["block_id"] == node].iloc[0]

        final_heat = get_hotspot_temp(
            mcpat_ptrace,
            float(mcpat_ptrace["execution_cycles"]) / float(clock_rate),
            config_file,
            new_heat,
        )

        heat_data[node] = final_heat

    return heat_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial ptrace files for each basic block"
    )
    parser.add_argument("--mcpat_outs", help="The path to the mcpat output directory")
    parser.add_argument(
        "--configs",
        type=str,
        default="./scripts/configs.cfg",
        help="General config file",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="",
        help="The prefix of every file in that output directory (default to be same as last level directiory name)",
    )
    parser.add_argument(
        "--module_index",
        type=int,
        default=2,
        help="The module index to include (expecting 2 usually for the profiled run)",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="hottest",
        help=f"Take `{AGGREGATE_METHOD_WORST_CASE}` temperature per unit, or `{AGGREGATE_METHOD_WEIGHTED_AVG}` average, or most `{AGGREGATE_METHOD_MOST_LIKELY}` ",
    )
    parser.add_argument(
        "--control_flow",
        type=str,
        default="CFG.csv",
        help="The control flow graph csv file",
    )
    parser.add_argument("--dag", type=str, default="DAG.csv", help="Path to DAG file")
    parser.add_argument(
        "--block_additional",
        type=str,
        default="PerBlockAdditional.csv",
        help="Path to per-block additional information file",
    )
    parser.add_argument(
        "--topo_sort",
        type=str,
        default="TopoComp.csv",
        help="Path to topologically sorted components file",
    )
    args = parser.parse_args()

    mcpat_df = load_folder_mcpat(args.mcpat_outs, args.file_prefix)
    config_data = utils.load_cfg(args.configs)
    cfg = utils.load_adjacency_list_cfg(args.control_flow, args.module_index)
    dag = utils.load_adjacency_list_dag(args.dag, args.module_index)
    topo = utils.load_topo_sort(args.topo_sort, args.module_index)
    additional_block = utils.load_block_additional(
        args.block_additional, args.module_index
    )

    # TODO: current goal, iterate over the the nodes, computing the final temperature, given the prev temperature(s)
    # 1. must have topologically sorted nodes
    # 2. must have mapping of node -> parents (reverse mapping)
    # 3. must be able to turn output temperature into initial temperature (unsure?)

    # NOTE: we cannot guarantee parents are always visited first, so sometimes we default to initial temperatures
    # 1. all SCC dependencies will be visited first
    # 2. within an SCC, we attempt to visit dependencies first, but can't always because of cycles
    #  - (TODO) prioritise in order of block id if not, since lower block id means higher in program means probably executed first

    # Component to nodes mapping (transforming df)
    comp_to_nodes_set = defaultdict(set)

    for _, row in additional_block.iterrows():
        comp_to_nodes_set[row["comp_id"]].add(row["block_id"])

    comp_to_nodes = defaultdict(list)

    for k, v in comp_to_nodes_set.items():
        comp_to_nodes[k] = list(v)

    # Global adjacency (transforming df)
    global_adj = defaultdict(list)

    for _, row in cfg.iterrows():
        global_adj[row["start_block_id"]].append(row["exit_block_id"])

    # Roughly topologically sorted nodes
    approx_sorted_nodes = ordered_nodes(topo, comp_to_nodes, global_adj)
    print(f"{approx_sorted_nodes=}")

    # mcpat_df: pd.DataFrame,
    # approx_sorted_nodes: list,
    # global_adj: dict,
    # additional_block: pd.DataFrame,
    # cfg: pd.DataFrame,
    # dag: pd.DataFrame,
    # aggregate_method: str,
    # clock_rate: float,
    # config_file: str,

    all_block_heats = calculate_all_heat(
        mcpat_df,
        approx_sorted_nodes,
        global_adj,
        additional_block,
        cfg,
        dag,
        args.aggregate,
        config_data["mcpat"]["CLOCK_RATE"],
        args.configs,
    )

    with open("HeatData.csv", "w") as f:
        f.write("block_id,")

        for i, col in enumerate(HOTSPOT_FLOORPLAN):
            if i > 0:
                f.write(",")

            f.write(col)

        f.write("\n")

        for block_id, heatmap in all_block_heats.items():
            f.write(f"{block_id},")

            for i, col in enumerate(HOTSPOT_FLOORPLAN):
                if i > 0:
                    f.write(",")

                res = heatmap.get(col, None)

                if res is None:
                    f.write("na")
                else:
                    f.write(f"{res:.3f}")

            f.write("\n")

    # TODO: output all block heats, or at least delta temp for a given path
    print(all_block_heats)


if __name__ == "__main__":
    main()
