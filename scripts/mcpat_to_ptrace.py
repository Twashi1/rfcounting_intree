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


def mcpat_to_hotspot_units(
    mcpat_out_file: str, flp_df: pd.DataFrame, use_residuals=True
) -> dict:
    unit_stats = utils.mcpat_to_dict(mcpat_out_file)

    # Unsure how to attribute most of these residuals, just doing the most impactful ones
    # Instruction Fetch Unit residual
    ifu_residual = utils.get_static_dynamic_power(
        unit_stats, ["Instruction Fetch Unit"]
    ) - utils.get_static_dynamic_power(
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
    renaming_residual = utils.get_static_dynamic_power(
        unit_stats, ["Renaming Unit"]
    ) - utils.get_static_dynamic_power(
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
    lsu_residual = utils.get_static_dynamic_power(
        unit_stats, ["Load Store Unit"]
    ) - utils.get_static_dynamic_power(
        unit_stats,
        [
            "Data Cache",
            "LoadQ",
            "StoreQ",
        ],
    )

    # Memory Management
    mmu_residual = utils.get_static_dynamic_power(
        unit_stats, ["Memory Management Unit"]
    ) - utils.get_static_dynamic_power(
        unit_stats,
        [
            "Itlb",
            "Dtlb",
        ],
    )
    # Execution Unit
    execution_residual = utils.get_static_dynamic_power(
        unit_stats, ["Execution Unit"]
    ) - utils.get_static_dynamic_power(
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
        flp_df["unit"].isin(set(execution_units)),
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

    l2_power = utils.get_static_dynamic_power(unit_stats, ["L2"])
    icache_power = utils.get_static_dynamic_power(unit_stats, ["Instruction Cache"])
    dcache_power = utils.get_static_dynamic_power(unit_stats, ["Data Cache"])
    bpred_power = utils.get_static_dynamic_power(unit_stats, ["Branch Predictor"])
    dtb_power = utils.get_static_dynamic_power(unit_stats, ["Dtlb"])
    fpu_power = utils.get_static_dynamic_power(unit_stats, ["Floating Point Unit"])
    fp_rf_power = utils.get_static_dynamic_power(unit_stats, ["Floating Point RF"])
    fp_map_power = utils.get_static_dynamic_power(
        unit_stats, ["FP Front End RAT", "FP Retire RAT", "FP Free List"]
    )
    int_map_power = utils.get_static_dynamic_power(
        unit_stats, ["Int Front End RAT", "Int Retire RAT", "Free List"]
    )
    intq_power = utils.get_static_dynamic_power(unit_stats, ["Instruction Window"])
    int_rf_power = utils.get_static_dynamic_power(unit_stats, ["Integer RF"])
    int_alu_power = utils.get_static_dynamic_power(unit_stats, ["Integer ALU"])
    fpq_power = utils.get_static_dynamic_power(unit_stats, ["FP Instruction Window"])
    lsq_power = utils.get_static_dynamic_power(unit_stats, ["LoadQ", "StoreQ"])
    itb_power = utils.get_static_dynamic_power(unit_stats, ["Itlb"])

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


# TODO: lots of code duplication between here and utils.load_folder_mcpat
def load_folder_mcpat_to_hotspot(
    folder_path: str, file_prefix: str, floormap_df: pd.DataFrame, use_residuals=True
) -> pd.DataFrame:
    folder_path = folder_path.rstrip(os.sep)
    last_directory_name = os.path.basename(folder_path)

    if file_prefix == "":
        file_prefix = last_directory_name

    if file_prefix == "":
        raise ValueError(f"Bad folder path? File prefix was empty. {folder_path=}")

    pattern = re.compile(rf"{re.escape(file_prefix)}_idx(\d{{4}})_v(\d+)\.txt$")

    all_data = defaultdict(list)

    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)

        if not os.path.isfile(file_path):
            continue

        hotspot_mappings = mcpat_to_hotspot_units(file_path, floormap_df, use_residuals)

        match = pattern.match(f)

        if not match:
            raise ValueError(f"Unrecognised file name format: {f}, in path {file_path}")

        path_number = int(match.group(1))
        voltage_level = int(match.group(2))

        all_data["voltage"].append(voltage_level)

        # TODO: we don't know if these are paths or blocks, so we have to add both
        all_data["path_index"].append(path_number)
        all_data["block_id"].append(path_number)

        for key, power in hotspot_mappings.items():
            all_data[key].append(power)

    return pd.DataFrame(all_data)


def get_hotspot_temp(
    hotspot_ptrace: dict,
    execution_time: float,
    config_file: str,
    hotspot_config_file: str,
    initial_heat: dict,
    heatsink_offset: float,
    name_id: str,
) -> dict:
    # TODO: construct input from hotspot_ptrace (dataframe series, or turn to dict?)
    # TODO: assume ttrace is a file path to use as input, initial_temp is numerical (kelvin)
    # TODO: call hotspot binary with the inputs
    # TODO: load outputs as ttrace output path?

    # Not sure how HotSpot works, so we're going to provide it 10 samples of the same power
    # With the sampling interval being 1/10th of the execution time
    # TODO: sample count should be selected such that we have intervals of span 0.3-3 microseconds (according to HotSpot)
    sample_count = 10

    # TODO: copy this file for logging
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

        for _ in range(sample_count):
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
        print(initial_heat)
        write_heatmap_to_init("./hotspot_files/gcc.init", initial_heat, heatsink_offset)

    # Replace line in example.config
    lines = []

    with open(hotspot_config_file, "r") as f:
        lines = f.readlines()

    pattern = re.compile(r"^\s*-sampling_intvl\s+[\d\.eE+-]+", re.MULTILINE)

    per_sample_execution_time = execution_time / sample_count
    per_sample_epsilon = 0.00001

    if not (per_sample_execution_time > per_sample_epsilon):
        # Just some epsilon value
        per_sample_execution_time = per_sample_epsilon

    # TODO: check whitespaces are correct
    # TODO: verify it correctly replaces the text
    for i, line in enumerate(lines):
        if pattern.match(line):
            # TODO:
            lines[i] = f"\t\t-sampling_intvl\t\t{per_sample_execution_time:.6f}\n"
            break

    with open(hotspot_config_file, "w") as f:
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
        prob = full_cfg.loc[
            (full_cfg["start_block_id"] == start_block_id)
            & (full_cfg["exit_block_id"] == target),
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


def write_heatmap_to_init(file_path: str, heat: dict, heatsink_offset: float) -> None:
    # Write values for each key
    # then write
    # we don't have any real values for the ones below? so we just guess them
    # TODO: there is almost surely a way to get the correct output such that we can give more valid inputs for these
    # - iface_
    # - hsp_
    # - hsink_
    # - inode_0 through inode_11 (unsure of correct values)
    print(f"Writing to {file_path}")
    temp = 75 + 273.15

    with open(file_path, "w") as f:
        for key, value in heat.items():
            f.write(f"{key} {value:.2f}\n")

        for key, value in heat.items():
            f.write(f"iface_{key} {value:.2f}\n")

        for key, value in heat.items():
            f.write(f"hsp_{key} {value:.2f}\n")

        for key, value in heat.items():
            heatsink_value = value + heatsink_offset
            f.write(f"hsink_{key} {heatsink_value:.2f}\n")

        # TODO: we have no clue of correct value, so we assume temperature of 75
        # (just assuming something pretty warm)
        for i in range(12):
            f.write(f"inode_{i} {temp:.2f}\n")


def read_heatmap_output(file_path: str) -> dict:
    # Reading a ttrace file (separated by space, regex to be safe)
    df = pd.read_csv(file_path, sep=r"\s+")
    # All columns numerical, so should work
    df = df.astype(float)

    # TODO: this is expected whenever we have >1 sample, not worth the error
    if len(df) > 1:
        print("[WARN] DF had more than one row, we just take the first")

    return df.iloc[-1].to_dict()


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
    hotspot_config_file: str,
    heatsink_offset: float,
):
    # Clock rate given in MHz, need Hz
    clock_rate = float(clock_rate)
    clock_rate *= 1.0e6

    heatsink_offset = float(heatsink_offset)

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

    for start in global_adj.keys():
        for end in global_adj[start]:
            parents[end].append(start)

    for node in approx_sorted_nodes:
        # Get all parents
        prevs = parents[node]
        # Get heat data of all parents
        parent_heats = {
            parent: heat_data[parent] for parent in prevs if parent in heat_data
        }

        print(
            f"Block id: {node=} has {len(parent_heats)} parents, or {prevs} dependencies"
        )

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
            hotspot_config_file,
            new_heat,
            heatsink_offset,
            f"{node:04d}",
        )

        heat_data[node] = final_heat

    return heat_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial ptrace files for each basic block"
    )
    parser.add_argument("--mcpat_outs", help="The path to the mcpat output directory")
    parser.add_argument(
        "--initial_temp_kelvin",
        type=float,
        default=273.15 + 77.0,
        help="Initial temperatures in kelvin when we cannot find previous block",
    )
    parser.add_argument(
        "--voltage_levels_file",
        type=str,
        default="VoltageLevels.csv",
        help="Voltage level to use per basic block",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="./scripts/configs.cfg",
        help="General config file",
    )
    parser.add_argument(
        "--configs_hotspot",
        type=str,
        default="./hotspot_files/example.config",
        help="Hotspot config file",
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
    parser.add_argument(
        "--voltage_levels",
        type=str,
        default="VoltageLevels.csv",
        help="Path to voltage levels file",
    )
    parser.add_argument(
        "--floorplan",
        type=str,
        default="./hotspot_files/ev6.flp",
        help="Hotspot floorplan",
    )
    args = parser.parse_args()

    flp_df = pd.read_csv(
        args.floorplan, delim_whitespace=True, header=None, index_col=0, comment="#"
    )

    flp_df.columns = ["width", "height", "leftx", "bottomy"]
    flp_df.index.name = "unit"
    flp_df = flp_df.reset_index()
    flp_df["width"] = flp_df["width"].astype(float)
    flp_df["height"] = flp_df["height"].astype(float)
    flp_df["leftx"] = flp_df["leftx"].astype(float)
    flp_df["bottomy"] = flp_df["bottomy"].astype(float)
    flp_df["area"] = flp_df["width"] * flp_df["height"]

    config_data = utils.load_cfg(args.configs)
    mcpat_df = load_folder_mcpat_to_hotspot(
        args.mcpat_outs,
        args.file_prefix,
        flp_df,
        config_data["hotspot"]["INCLUDE_RESIDUALS"] == "true",
    )
    cfg = utils.load_adjacency_list_cfg(args.control_flow, args.module_index)
    dag = utils.load_adjacency_list_dag(args.dag, args.module_index)
    topo = utils.load_topo_sort(args.topo_sort, args.module_index)
    additional_block = utils.load_block_additional(
        args.block_additional, args.module_index
    )
    voltage_levels = utils.load_voltage_levels(args.voltage_levels_file)

    mcpat_df_filtered = mcpat_df.loc[
        mcpat_df.set_index(["block_id", "voltage"]).index.isin(
            voltage_levels.set_index(["block_id", "voltage_level"]).index
        )
    ]

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
        mcpat_df_filtered,
        approx_sorted_nodes,
        global_adj,
        additional_block,
        cfg,
        dag,
        args.aggregate,
        config_data["mcpat"]["CLOCK_RATE"],
        args.configs,
        args.configs_hotspot,
        config_data["hotspot"]["HEATSINK_OFFSET"],
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


if __name__ == "__main__":
    main()
