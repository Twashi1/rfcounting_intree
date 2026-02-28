import utils
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Given heat data, calculate the required voltage to maintain the target frequency"
    )
    parser.add_argument("--program_heat", type=str, help="Heat data file")
    parser.add_argument(
        "--config_file",
        type=str,
        default="./scripts/configs.cfg",
        help="Config file to get the voltages and target clock frequency from",
    )
    parser.add_argument(
        "--module_index",
        type=int,
        default=2,
        help="The module to get for block additional data",
    )
    parser.add_argument(
        "--out_prefix", type=str, help="Name of the program/prefix to output under"
    )
    args = parser.parse_args()

    # TODO: this approach is very limited
    # - changing the voltage changes the temperature, thus we need to consider the full CFG
    # - changing the temperature from above also changes the required voltage

    # We can run this, change VoltageLevels.csv accordingly, and re-run, however
    # we only compute power traces for some subset of the possible voltages
    # (we can run it for all in a more final test, however would take much longer)
    # This means that when we update the voltage, we might not be able to get a new, accurate temperature prediction
    # since the McPAT power trace only exists for some voltages

    # Load per-basic block heat data
    heat_df = utils.load_program_heats(args.program_heat)
    # Load all possible voltage levels
    cfg = utils.load_cfg(args.config_file)
    clock_frequency = int(cfg["mcpat"]["CLOCK_RATE"])
    clock_frequency_ghz = float(clock_frequency) / 1_000.0
    voltage_levels = utils.load_voltage_levels_from_cfg(cfg)
    # Find required voltage for each basic block
    # TODO: should do this vectorised?
    per_block_required_voltage_levels = []

    for _, row in heat_df.iterrows():
        # Convert to celsius
        est_temp = float(row["temp_max"]) - 273.15
        required_voltage = utils.tei_select_voltage(
            cfg, est_temp, clock_frequency_ghz, voltage_levels
        )

        per_block_required_voltage_levels.append(required_voltage)

    # Output that as a table
    heat_df["required_voltage_value"] = per_block_required_voltage_levels
    heat_df.to_csv(f"{args.out_prefix}_ProgramHeatVoltages.csv", index=False)


if __name__ == "__main__":
    main()
