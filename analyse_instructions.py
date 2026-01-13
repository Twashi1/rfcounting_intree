# TODO: given output from McPAT
# TODO: given input XML
# compare power to number of int instructions, and number of float instructions (not perfect solution)
import argparse
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import nnls 

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    int_node = root.find(".//component[@id='system.core0']/stat[@name='int_instructions']")
    float_node = root.find(".//component[@id='system.core0']/stat[@name='fp_instructions']")

    int_instructions = int(int_node.get("value"))
    fp_instructions = int(float_node.get("value"))

    return int_instructions, fp_instructions 

def parse_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()

    seen = False

    for line in lines:
        if not seen:
            if "Data Cache" in line:
                seen = True

        else:
            m = re.search(r"Runtime Dynamic\s*=\s*([\d.]+)", line)
            if m:
                return float(m.group(1))

    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input file with pairs XML_file McPAT_output_file")
    args = p.parse_args()

    with open(args.input) as f:
        X = []
        Y = []

        for line in f:
            xml_path, txt_path = line.strip().split()

            int_instructions, fp_instructions = parse_xml(xml_path)
            power = parse_txt(txt_path)

            X.append([int_instructions, fp_instructions])
            Y.append(power)

            print(xml_path, txt_path, int_instructions, fp_instructions, power)

        X = np.array(X)
        Y = np.array(Y)

        coef, _ = nnls(X, Y)
        
        print(f"Estimated runtime dynamic power per int_instruction: {coef[0]}")
        print(f"Estimated runtime dynamic power per fp_instruction: {coef[1]}")

if __name__ == "__main__":
    main()
