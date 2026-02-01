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


# TODO: "name" is actually name prefix
def mcpat_get_unit_stats(name: str, text: str) -> dict:
    # Note sometimes a colon, sometimes not
    pattern = rf"\s*{re.escape(name)}.*\n(\s+[\s\w=]+\s*([\d\.])+.*\n)+"

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
        fm = re.search(rf"{key}\s*=\s*([0-9.eE+-]+)", block)
        if fm:
            fields[key] = float(fm.group(1))

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
        all_data["path_index"].append(path_number)

        for key, power in unit_stats.items():
            all_data[key].append(power)

    return pd.DataFrame(all_data)


def get_hotspot_temp(
    hotspot_ptrace: dict,
    execution_time: float,
    config_file: str,
    initial_ttrace: str = None,
) -> None:
    # TODO: construct input from hotspot_ptrace (dataframe series, or turn to dict?)
    # TODO: assume ttrace is a file path to use as input, initial_temp is numerical (kelvin)
    # TODO: call hotspot binary with the inputs
    # TODO: load outputs as ttrace output path?
    with open("./hotspot_files/gcc.ptrace", "w") as f:
        # Write in keys
        for i, key in enumerate(hotspot_ptrace.keys()):
            if i > 0:
                f.write(" ")

            f.write(key)

        f.write("\n")

        # Write in values
        for i, key in enumerate(hotspot_ptrace.keys()):
            if i > 0:
                f.write(" ")

            f.write(hotspot_ptrace[key])

        f.write("\n")

    if initial_ttrace is None:
        # TODO: disable gcc.init somehow
        pass
    else:
        shutil.copyfile(initial_ttrace, "./hotspot_files/gcc.init")

    # Replace line in example.config
    lines = []

    with open(config_file, "r") as f:
        lines = f.readlines()

    pattern = re.compile("^\s*sampling_intvl\s+[\d.eE+-]+")

    # TODO: check whitespaces are correct
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"\t-sampling_intvl\t\t{execution_time}\n"
            break

    with open(config_file, "w") as f:
        f.writelines(lines)

    # Run hotspot
    should_use_initial = "true" if initial_ttrace is not None else "false"
    subprocess.run(["./run_hotspot.sh", should_use_initial], check=True)


def ordered_nodes(scc_topo: list, comp_to_nodes: dict, adj: pd.DataFrame):
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

        topo_sorted = []

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
    filtered = [node for node in nodes if node heat_data]
    # Get heat data of filtered nodes
    heats = [heat_data[node] for node in filtered]
    # Compute branch probabilities for each edge (used in aggregates)
    branch_prob = []

    for start_block_id in filtered:
        prob = df.loc[
            (df["start_block_id"] == start_block_id) &
            (df["exit_block_id"] == target),
            "branch_prob"
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

def calculate_all_heat(
    approx_sorted_nodes: list,
    global_adj: dict,
    additional_block: pd.DataFrame,
    cfg: pd.DataFrame,
    dag: pd.DataFrame,
    aggregate_method: str,
):
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

    # TODO: load heat output as heat data
    # TODO: write heat data as heat input
    pass


def main():
    # TODO: assume we have some CFG data with branch probability info, and block IDs that map exactly
    #   to those in the DVS calling points we were given
    # TODO: current input files just give us IDs in some arbitrary order
    #   - need some format like _pathIndex_blockID_dvsCallingPointIndex
    #   - then we can easily extract per-block ID branches, and determine what should be
    #       1. all possible previous ptraces (temps)
    #       2. either weighted average, take worst-case, take most likely of these ptraces
    #     and use that as our initial temperatures
    unit_stats = mcpat_to_dict("../mcpat_out/gemm/gemm_idx0000_low.txt")
    # print(unit_stats)

    # DataFrame has voltage level, path index, and power stats per hotspot component
    df = load_folder_mcpat("../mcpat_out/gemm", "")
    print(df)

    parser = argparse.ArgumentParser(
        description="Generate initial ptrace files for each basic block"
    )
    parser.add_argument("--mcpat_outs", help="The path to the mcpat output directory")
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

    df = load_folder_mcpat(args.mcpat_outs, args.file_prefix)
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
    comp_to_nodes = defaultdict(list)

    for _, row in dag.iterrows():
        comp_to_nodes[row["comp_id"]].append(row["block_id"])

    # Global adjacency (transforming df)
    global_adj = defaultdict(list)

    for _, row in cfg.iterrows():
        global_adj[row["start_block_id"]].append(row["exit_block_id"])

    # Roughly topologically sorted nodes
    approx_sorted_nodes = ordered_nodes(topo, comp_to_nodes, global_adj)
    calculate_all_heat(approx_sorted_nodes, global_adj, additional_block, cfg, dag)

    # TODO: correctly use get_hotspot_temp function


if __name__ == "__main__":
    main()
