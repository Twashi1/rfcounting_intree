import utils
import mcpat_to_ptrace as mcpt
import pandas as pd

PROGRAMS = [
    "2mm",
    "3mm",
    "adi",
    "atax",
    "covariance",
    "durbin",
    "floyd-warshall",
    "gemm",
    "gemver",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "symm",
    "syrk",
]


def pe(actual, obtained):
    return abs(obtained - actual) / actual * 100


config = utils.load_cfg("./scripts/configs.cfg")


flp_df = pd.read_csv(
    "./hotspot_files/ev6.flp",
    delim_whitespace=True,
    header=None,
    index_col=0,
    comment="#",
)

flp_df.columns = ["width", "height", "leftx", "bottomy"]
flp_df.index.name = "unit"
flp_df = flp_df.reset_index()
flp_df["width"] = flp_df["width"].astype(float)
flp_df["height"] = flp_df["height"].astype(float)
flp_df["leftx"] = flp_df["leftx"].astype(float)
flp_df["bottomy"] = flp_df["bottomy"].astype(float)
flp_df["area"] = flp_df["width"] * flp_df["height"]

output_queue = []

for program in PROGRAMS:
    dir = f"./gem5_stats/{program}/stats.txt"
    utils.create_standard_stat_file(dir, f"./stats/{program}_gem5")
    gem5_stats = utils.load_standard_stat_file(f"./stats/{program}_gem5_STD.csv")
    our_stats = utils.load_standard_stat_file(f"./stats/{program}_path_STD.csv")
    # TODO: need to generate a bunch of extra data through scripts; annoying

    cycle_count = our_stats[utils.CYCLE_COUNT].sum()
    int_instr = our_stats[utils.INT_INSTR_COUNT].sum()
    float_instr = our_stats[utils.FLOAT_INSTR_COUNT].sum()
    instr_count = our_stats[utils.INSTR_COUNT].sum()

    gem5_cycle_count = gem5_stats[utils.BUSY_CYCLES].sum()
    gem5_int_instr = gem5_stats[utils.INT_INSTR_COUNT].sum()
    gem5_float_instr = gem5_stats[utils.FLOAT_INSTR_COUNT].sum()
    gem5_instr_count = gem5_stats[utils.INSTR_COUNT].sum()

    cycle_error = pe(gem5_cycle_count, cycle_count)
    int_error = pe(gem5_int_instr, int_instr)
    float_error = pe(gem5_float_instr, float_instr)
    inst_error = pe(gem5_instr_count, instr_count)

    output_queue.append(f"Program: {program}")
    output_queue.append(f"Cycle error {cycle_error:.2f}%")
    output_queue.append(f"Int error {int_error:.2f}%")
    output_queue.append(f"Float error {float_error:.2f}%")
    output_queue.append(f"Instr error {inst_error:.2f}%")
    output_queue.append(f"Rel Cycle error {cycle_count - gem5_cycle_count}")
    output_queue.append(f"Rel Int error {int_instr - gem5_int_instr:.2f}")
    output_queue.append(f"Rel Float error {float_instr - gem5_float_instr:.2f}")
    output_queue.append(f"Rel Instr error {instr_count - gem5_instr_count:.2f}")

    numerical_sums = our_stats.select_dtypes(include="number").sum()
    # Get first of all other columns
    other = our_stats.drop(columns=numerical_sums.index).iloc[0]
    stats = numerical_sums.combine_first(other)
    # Stats isn't strictly a dataframe at this point, but it works fine anyway
    our_stats = stats.to_frame().T

    gem5_stats["block_id"] = 999
    our_stats["block_id"] = 999

    gem5_trace_request = utils.PowerTraceRequestSpec(
        999,
        0.8,
        [0.6, 0.65, 0.7, 0.75, 0.8],
        f"./mcpat_inputs/{program}_gem5",
        f"./mcpat_out/{program}_gem5",
        f"{program}_gem5",
        gem5_stats,
        "./mcpat_inputs/Alpha21364.xml",
        3.0,
        config,
    )

    our_trace_request = utils.PowerTraceRequestSpec(
        999,
        0.8,
        [0.6, 0.65, 0.7, 0.75, 0.8],
        f"./mcpat_inputs/{program}_our",
        f"./mcpat_out/{program}_our",
        f"{program}_our",
        our_stats,
        "./mcpat_inputs/Alpha21364.xml",
        3.0,
        config,
    )

    gem5_power = utils.request_power_for_specification(gem5_trace_request)
    our_power = utils.request_power_for_specification(our_trace_request)

    core_power_gem5 = utils.get_static_dynamic_power(gem5_power, ["Core"])
    core_power_our = utils.get_static_dynamic_power(our_power, ["Core"])

    hotspot_ptrace_gem5 = utils.mcpat_to_hotspot_units(gem5_power, flp_df, True)
    hotspot_ptrace_our = utils.mcpat_to_hotspot_units(our_power, flp_df, True)

    final_heat_gem5 = mcpt.get_hotspot_temp(
        hotspot_ptrace_gem5,
        float(gem5_cycle_count) / float(3.0 * 1.0e9),
        config,
        "./hotspot_files/example.config",
        None,
        2,
        3.0,
        f"0999_3.0Hz",
    )

    final_heat_our = mcpt.get_hotspot_temp(
        hotspot_ptrace_our,
        float(cycle_count) / float(3.0 * 1.0e9),
        config,
        "./hotspot_files/example.config",
        None,
        2,
        3.0,
        f"0999_3.0Hz",
    )

    output_queue.append("Gem5 final heat:")
    output_queue.append(final_heat_gem5)
    output_queue.append("Our final heat:")
    output_queue.append(final_heat_our)

    max_temp_gem5 = max(final_heat_gem5.values())
    max_temp_our = max(final_heat_our.values())

    avg_temp_gem5 = sum(final_heat_gem5.values()) / len(final_heat_gem5)
    avg_temp_our = sum(final_heat_our.values()) / len(final_heat_our)

    max_temp_error = pe(max_temp_gem5, max_temp_our)
    avg_temp_error = pe(avg_temp_gem5, avg_temp_our)

    output_queue.append(f"Max Temperature percent error: {max_temp_error:.2f}%")
    output_queue.append(
        f"Max Temperature objective error: {abs(max_temp_gem5 - max_temp_our)}"
    )
    output_queue.append(f"Avg Temperature percent error: {avg_temp_error:.2f}%")
    output_queue.append(
        f"Avg Temperature objective error: {abs(avg_temp_gem5 - avg_temp_our)}"
    )

with open("final_accuracy.txt", "w") as f:
    for item in output_queue:
        print(item)
        f.write(f"{item}\n")
