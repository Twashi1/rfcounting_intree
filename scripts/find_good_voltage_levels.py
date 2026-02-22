import utils


def main():
    min_voltage = 0.6
    max_voltage = 0.9
    num_steps = 60
    step_size = (max_voltage - min_voltage) / float(num_steps)
    voltage_dict = {i: (min_voltage + step_size * i) for i in range(num_steps)}

    for temp in range(60, 88, 2):
        required_voltages = utils.get_voltage(temp, 3.0, voltage_dict)
        actual_voltage = voltage_dict[required_voltages]

        print(f"At temperature {temp}, we need voltage {actual_voltage}")


if __name__ == "__main__":
    main()
