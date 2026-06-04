import pandas as pd
import utils
import os

def generate_gem5_sheet():
    gem5_runs = [
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
        "syrk"
    ]

    os.makedirs("./excel", exist_ok=True)
    combined_stats = []

    for benchmark_name in gem5_runs:
        input_file = f"./gem5_stats/{benchmark_name}/stats.txt"
        stats_blocks = utils.get_stats_df_gem5_run(input_file)
        stat_list = stats_blocks.to_dict(orient="records")

        assert len(stat_list) == 1

        stats = stat_list[0]
        stats["benchmark"] = benchmark_name

        combined_stats.append(stats)

    df = pd.DataFrame(combined_stats)
    df = df.set_index("benchmark")

    output_path = f"./excel/gem5_runs.xlsx"
    df.to_excel(output_path, index=True)

def generate_efficiency_stats():
    res = utils.load_efficiency_stats("efficiencyStats.txt")

    all_stats = []

    for benchmark, stats in res.items():
        stats["benchmark"] = benchmark
        all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    df = df.set_index("benchmark")

    output_path = f"./excel/efficiency_stats.xlsx"
    df.to_excel(output_path, index=True)

def main():
    generate_gem5_sheet()
    generate_efficiency_stats()

if __name__ == "__main__":
    main()
