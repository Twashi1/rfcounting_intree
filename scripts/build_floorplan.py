import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import utils


def random_color():
    hue = random.random()
    saturation = 0.9
    value = 0.7

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)


# L2_left	0.004900	0.006200	0.000000	0.009800
# L2	0.016000	0.009800	0.000000	0.000000
# L2_right	0.004900	0.006200	0.011100	0.009800
# Icache	0.003100	0.002600	0.004900	0.009800
# Dcache	0.003100	0.002600	0.008000	0.009800
# Bpred_0	0.001033	0.000700	0.004900	0.012400
# Bpred_1	0.001033	0.000700	0.005933	0.012400
# Bpred_2	0.001033	0.000700	0.006967	0.012400
# DTB_0	0.001033	0.000700	0.008000	0.012400
# DTB_1	0.001033	0.000700	0.009033	0.012400
# DTB_2	0.001033	0.000700	0.010067	0.012400
# FPAdd_0	0.001100	0.000900	0.004900	0.013100
# FPAdd_1	0.001100	0.000900	0.006000	0.013100
# FPReg_0	0.000550	0.000380	0.004900	0.014000
# FPReg_1	0.000550	0.000380	0.005450	0.014000
# FPReg_2	0.000550	0.000380	0.006000	0.014000
# FPReg_3	0.000550	0.000380	0.006550	0.014000
# FPMul_0	0.001100	0.000950	0.004900	0.014380
# FPMul_1	0.001100	0.000950	0.006000	0.014380
# FPMap_0	0.001100	0.000670	0.004900	0.015330
# FPMap_1	0.001100	0.000670	0.006000	0.015330
# IntMap	0.000900	0.001350	0.007100	0.014650
# IntQ	0.001300	0.001350	0.008000	0.014650
# IntReg_0	0.000900	0.000670	0.009300	0.015330
# IntReg_1	0.000900	0.000670	0.010200	0.015330
# IntExec	0.001800	0.002230	0.009300	0.013100
# FPQ	0.000900	0.001550	0.007100	0.013100
# LdStQ	0.001300	0.000950	0.008000	0.013700
# ITB_0	0.000650	0.000600	0.008000	0.013100
# ITB_1	0.000650	0.000600	0.008650	0.013100


def parse_floorplan(file: str):
    units = {}

    with open(file) as f:
        for line in f.readlines():
            if line.startswith("#") or len(line) == 1:
                continue

            unit_name, w, h, x, y = line.split("\t")

            units[unit_name] = {
                "w": float(w.strip()),
                "h": float(h.strip()),
                "x": float(x.strip()),
                "y": float(y.strip()),
            }

    return units


def draw_all_units(units: dict):
    _fig, ax = plt.subplots(figsize=(16, 8))
    ax: Axes

    # Determine plot bounds
    min_x = min(u["x"] for u in units.values())
    min_y = min(u["y"] for u in units.values())
    max_x = max(u["x"] + u["w"] for u in units.values())
    max_y = max(u["y"] + u["h"] for u in units.values())

    for name, u in units.items():
        x = u["x"]
        y = u["y"]
        w = u["w"]
        h = u["h"]

        color = random_color()

        rect = Rectangle(
            (x, y),
            w,
            h,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.6,
        )
        ax.add_patch(rect)

        ax.text(
            x + w / 2,
            y + h / 2,
            name,
            ha="center",
            va="center",
            fontsize=6,
            color="black",
        )

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect(0.5)  # compress the y axis so the names appear clearer
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Unit Layout")

    plt.show()


def floorplan_for_program():
    # TODO: we've manually laoded these values, based on 14nm in config
    # Map from area of McPAT unit to area of HotSpot unit
    # Units are mm^2
    # NOTE: ev6 processor expects metres
    l2_area = 2.15867
    icache_area = 0.120769
    dcache_area = 0.184539
    bpred_area = 0.0285988
    dtb_area = 0.0227877
    itb_area = 0.00792202
    fpu_area = 0.0151602
    fprf_area = 0.0169824
    fpmap_area = 0.00299835 + 0.000338574 + 0.00101548
    intmap_area = 0.00933301 + 0.00219689 + 0.000920566
    intq_area = 0.00431003
    intrf_area = 0.0371817
    intalu_area = 0.002904 * 4  # TODO: unsure if we should be multiplying by count here
    fpq_area = 0.000895281
    lsq_area = 0.00428258 + 0.00428258

    ifu_residual_area = (
        0.202495 - 0.120769 - 0.0501103 - 0.0285988 - 0.00074008 - 0.000855912
    )
    renaming_residual_area = 0.0182242 - fpmap_area - intmap_area
    lsu_residual_area = 0.194526 - lsq_area
    mmu_residual_area = 0.0321311 - itb_area - dtb_area
    # TODO: unsure on the multiply 4 for Int ALU cores
    execution_residual_area = (
        0.0969234 - 0.0541641 - 0.0230592 - 0.002904 * 4 - 0.0151602 - 0.000214464
    )

    # TODO: some of these conversions are a bit lazy, should instead distribute according to what fraction of area the hotspot unit makes up
    # TODO: residuals
    hotspot_units = {
        "l2_left": l2_area / 3.0,
        "l2": l2_area / 3.0,
        "l2_right": l2_area / 3.0,
        "icache": icache_area,
        "dcache": dcache_area,
        "bpred_0": bpred_area / 3.0,
        "bpred_1": bpred_area / 3.0,
        "bpred_2": bpred_area / 3.0,
        "dtb_0": dtb_area / 3.0,
        "dtb_1": dtb_area / 3.0,
        "dtb_2": dtb_area / 3.0,
        # NOTE: dividing by 4 because averaging fpu over 2 adds and 2 muls
        "fpadd_0": fpu_area / 4.0,
        "fpadd_1": fpu_area / 4.0,
        "fpmul_0": fpu_area / 4.0,
        "fpmul_1": fpu_area / 4.0,
        "fpreg_0": fprf_area / 4.0,
        "fpreg_1": fprf_area / 4.0,
        "fpreg_2": fprf_area / 4.0,
        "fpreg_3": fprf_area / 4.0,
        "fpmap_0": fpmap_area / 2.0,
        "fpmap_1": fpmap_area / 2.0,
        "intmap": intmap_area,
        "intq": intq_area,
        "intreg_0": intrf_area / 2.0,
        "intreg_1": intrf_area / 2.0,
        "intexec": intalu_area,
        "fpq": fpq_area,
        "ldstq": lsq_area,
        "itb_0": itb_area / 2.0,
        "itb_1": itb_area / 2.0,
    }

    # Convert to metres squared (1e-3 * 1e-3)
    hotspot_units_wh_metres = {}

    for unit_name, value in hotspot_units.items():
        metres_squared = value * 1e-6
        length = metres_squared**0.5
        hotspot_units_wh_metres[unit_name] = {
            "w": length,
            "h": length,
            "x": 0.0,
            "y": 0.0,
        }

    hotspot_new_positions = compute_all_positions(hotspot_units_wh_metres)

    draw_all_units(hotspot_new_positions)

    # TODO: write out to file (with capitalisation restored...)
    return hotspot_new_positions


def compute_all_positions(data: dict):
    """
    Given areas of each unit, calculate the x, y positions

    NOTE: hardcoded based on observing the floorplan from drawing
    """
    top = lambda unit: data[unit]["y"] + data[unit]["h"]
    right = lambda unit: data[unit]["x"] + data[unit]["w"]

    data["l2_left"]["x"] = 0.0
    data["l2_left"]["y"] = top("l2")

    data["l2"]["x"] = 0.0
    data["l2"]["y"] = 0.0

    data["icache"]["x"] = right("l2_left")
    data["icache"]["y"] = top("l2")

    data["bpred_0"]["x"] = right("l2_left")
    data["bpred_1"]["x"] = right("bpred_0")
    data["bpred_2"]["x"] = right("bpred_1")

    data["bpred_0"]["y"] = top("icache")
    data["bpred_1"]["y"] = top("icache")
    data["bpred_2"]["y"] = top("icache")

    bpred_top = max(map(top, ("bpred_0", "bpred_1", "bpred_2")))

    data["fpadd_0"]["x"] = right("l2_left")
    data["fpadd_1"]["x"] = right("fpadd_0")
    data["fpadd_0"]["y"] = bpred_top
    data["fpadd_1"]["y"] = bpred_top

    fpadd_top = max(map(top, ("fpadd_0", "fpadd_1")))

    data["fpreg_0"]["x"] = right("l2_left")
    data["fpreg_1"]["x"] = right("fpreg_0")
    data["fpreg_2"]["x"] = right("fpreg_1")
    data["fpreg_3"]["x"] = right("fpreg_2")
    data["fpreg_0"]["y"] = fpadd_top
    data["fpreg_1"]["y"] = fpadd_top
    data["fpreg_2"]["y"] = fpadd_top
    data["fpreg_3"]["y"] = fpadd_top

    fpreg_top = max(map(top, ("fpreg_0", "fpreg_1", "fpreg_2")))

    data["fpmul_0"]["x"] = right("l2_left")
    data["fpmul_1"]["x"] = right("fpmul_0")
    data["fpmul_0"]["y"] = fpreg_top
    data["fpmul_1"]["y"] = fpreg_top

    fpmul_top = max(map(top, ("fpmul_0", "fpmul_1")))

    data["fpmap_0"]["x"] = right("l2_left")
    data["fpmap_1"]["x"] = right("fpmap_0")
    data["fpmap_0"]["y"] = fpmul_top
    data["fpmap_1"]["y"] = fpmul_top

    fp_small_stuff_right = max(map(right, ("fpmap_1", "fpmul_1", "fpreg_3", "fpadd_1")))

    data["fpq"]["x"] = fp_small_stuff_right
    data["fpq"]["y"] = bpred_top

    data["intmap"]["x"] = fp_small_stuff_right
    data["intmap"]["y"] = top("fpq")

    # Moving onto the right half
    left_half_right_edge = max(map(right, ("intmap", "fpq", "bpred_2", "icache")))

    data["dcache"]["x"] = left_half_right_edge
    data["dcache"]["y"] = top("l2")

    data["dtb_0"]["x"] = left_half_right_edge
    data["dtb_1"]["x"] = right("dtb_0")
    data["dtb_2"]["x"] = right("dtb_1")
    data["dtb_0"]["y"] = top("dcache")
    data["dtb_1"]["y"] = top("dcache")
    data["dtb_2"]["y"] = top("dcache")

    dtb_top = max(map(top, ("dtb_0", "dtb_1", "dtb_2")))

    data["itb_0"]["x"] = left_half_right_edge
    data["itb_1"]["x"] = right("itb_0")
    data["itb_0"]["y"] = dtb_top
    data["itb_1"]["y"] = dtb_top

    itb_top = max(map(top, ("itb_0", "itb_1")))

    data["ldstq"]["x"] = left_half_right_edge
    data["ldstq"]["y"] = itb_top

    data["intq"]["x"] = left_half_right_edge
    data["intq"]["y"] = top("ldstq")

    queues_right_edge = max(map(right, ("intq", "ldstq", "itb_1")))

    data["intexec"]["x"] = queues_right_edge
    data["intexec"]["y"] = dtb_top

    data["intreg_0"]["x"] = queues_right_edge
    data["intreg_1"]["x"] = right("intreg_0")
    data["intreg_0"]["y"] = top("intexec")
    data["intreg_1"]["y"] = top("intexec")

    int_right_edge = max(map(right, ("dcache", "dtb_2", "intexec", "intreg_1")))

    data["l2_right"]["x"] = int_right_edge
    data["l2_right"]["y"] = top("l2")

    return data


def write_out_floorplan(filename: str, flp: dict, original: dict):
    unmap_names = {name.lower(): name for name in original.keys()}

    unmapped_flp = {}

    for name in flp:
        unmapped_flp[unmap_names[name.lower()]] = flp[name]

    with open(filename, "w") as f:
        for name, value in unmapped_flp.items():
            x = value["x"]
            y = value["y"]
            w = value["w"]
            h = value["h"]
            f.write(f"{name}\t{w:.8f}\t{h:.8f}\t{x:.8f}\t{y:.8f}\n")


def main():
    units = parse_floorplan("./hotspot_files/ev6.flp")
    # draw_all_units(units)
    flp = floorplan_for_program()
    write_out_floorplan("./hotspot_files/out_flp.flp", flp, units)


if __name__ == "__main__":
    main()
