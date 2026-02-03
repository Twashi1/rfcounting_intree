import argparse
import utils
import pandas as pd

# NOTE: we removed L1 caches, L2 cache, Translation Lookaside Buffers
# TODO: maybe some more of these should be excluded? like the queues IntQ, FPQ, LdStQ?
HOTSPOT_CORE_FLOORPLAN = {
    "Bpred_0",
    "Bpred_1",
    "Bpred_2",
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
}


def main():
    # TODO: load HeatData.csv
    # TODO: load output if exists
    # TODO: get max of functional units, get mean of functional units, get weighted on width/height
    # TODO:
    parser = argparse.ArgumentParser(
        description="Get average, and max heat data per functional unit"
    )
    parser.add_argument(
        "--heatdata", type=str, default="HeatData.csv", help="Heat data file"
    )
    parser.add_argument(
        "--floorplan",
        type=str,
        default="./hotspot_files/ev6.flp",
        help="Hotspot floorplan",
    )
    parser.add_argument(
        "--additional_block",
        type=str,
        default="PerBlockAdditional.csv",
        help="Additional block data for execution time",
    )
    parser.add_argument(
        "--module_index",
        type=int,
        default=2,
        help="The module to get for block additional data",
    )
    parser.add_argument(
        "--name", type=str, help="Name of the program/prefix to output under"
    )
    args = parser.parse_args()

    block_additional = utils.load_block_additional(
        args.additional_block, args.module_index
    )

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

    print(flp_df)

    df = pd.read_csv(args.heatdata, na_values=["nan", "NaN", ""])
    df = df.apply(pd.to_numeric, errors="coerce")
    df["block_id"] = df["block_id"].astype(int)

    core_cols = list(HOTSPOT_CORE_FLOORPLAN)

    df["temp_mean"] = df[core_cols].mean(axis=1)
    df["temp_max"] = df[core_cols].max(axis=1)

    areas = flp_df.set_index("unit")["area"]

    df["temp_area_weighted_mean"] = (
        df[core_cols].mul(areas[core_cols], axis=1).sum(axis=1) / areas[core_cols].sum()
    )

    df = df.merge(block_additional, on="block_id", how="inner")

    df.to_csv(f"{args.name}_ProgramHeat.csv", index=True)


if __name__ == "__main__":
    main()
