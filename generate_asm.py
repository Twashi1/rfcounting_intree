import argparse
import random
import pathlib

integer_ops = [
    "x += ({imm})",
    "x -= ({imm})",
    "x *= ({imm})",
    "x /= ({imm}|1)", # |1 for division by 0
    "x %= ({imm}|1)",
    "x <<= ({sh})",
    "x >>= ({sh})",
    "x ^= y",
    "x &= y",
    "x |= y",
    "if (x & {mask}) x += y ^ {imm}; x -= y & {imm}",
    "x++",
    "x--",
    "y = y + (x & {mask})",
    "z = z + static_cast<std::int64_t>(x - y)",
    "x = ~x",
    "x = x * x + y",
    "switch (x & 7) {{ case 0: x += {imm}; break; case 1: x -= {imm}; break; case 2: x *= {imm}; break; default: x ^= {imm}; break; }}"
]

# TODO: maybe consider sqrtf or log? these probably would generate a bunch of floating ops, or have a dedicated op 
floating_ops = [
    "x += {imm}",
    "x -= {imm}",
    "x *= {imm}",
    "x = {imm} != 0.0 ? x / {imm} : 0.0",
    "y = y + x * x",
    "z = (x + y) * (x - y)",
    "if (z > 0.0) z = x + y; else z = x -y",
    "x = static_cast<double>(x > y)",
    "x = static_cast<double>(x < y)",
    "x = static_cast<double>(x == y)",
    "x = x * x"
]

def rand_int_literal(signed, num_bytes):
    lb = -(2 ** (num_bytes - 1)) if signed else 0
    ub = 2 ** (num_bytes - 1) - 1 if signed else (2 ** num_bytes) - 1
    v = random.randint(lb, ub)

    suffix = "LL" if signed else "ULL"

    if v < 0:
        return f"-0x{-v:X}{suffix}"

    return f"0x{v:X}{suffix}"

def rand_float_literal(num_bytes):
    if num_bytes != 32 and num_bytes != 64: return "0.0f";

    # 0-1 value
    v = random.random()

    if num_bytes == 32:
        return f"{v:.8f}f"

    return f"{v:.16f}"

config_int = {
    "main_type": "std::uint64_t",
    "second_type": "std::int64_t",
    "main_init_literal": "0",
    "second_init_literal": "0",
    "main_random_literal": lambda: rand_int_literal(True, 64),
    "second_random_literal": lambda: rand_int_literal(False, 64),
    "rand_imm": lambda: random.randint(1, 0xFF)
}

config_float = {
    "main_type": "float",
    "second_type": "float",
    "main_init_literal": "0.0f",
    "second_init_literal": "0.0f",
    "main_random_literal": lambda: rand_float_literal(32),
    "second_random_literal": lambda: rand_float_literal(32),
    "rand_imm": lambda: random.random()
}

# the double curlies are for the python formatter
program_template = """#include <cstdint>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>

#if defined(_MSC_VER)
    #define NOINLINE __declspec(noinline)
#else
    #define NOINLINE __attribute__((noinline))
#endif

// volatile to prevent full optimisations
volatile std::uint64_t g_acc0 = 0;
volatile {main_type} g_acc1 = {main_init_literal};
volatile {second_type} g_signed = {second_init_literal};

NOINLINE void save_values_codegen(volatile {main_type} x, volatile {main_type} y, volatile {second_type} z) {{
    // TODO: this is bad, just use a different operation
    g_acc0 ^= *reinterpret_cast<volatile std::uint64_t*>(&x);
    g_acc1 += y;
    g_signed += z;
}}

NOINLINE void block_wrap(volatile {main_type}& x, volatile {main_type}& y, volatile {second_type}& z) {{
    {body}
}}

int main(int argc, char** argv) {{
    unsigned seed = static_cast<unsigned>(time(nullptr));

    std::srand(seed);

    // random values
    volatile {main_type} a = {l0};
    volatile {main_type} b = {l1};
    volatile {second_type} c = {l2};

    block_wrap(a, b, c);

    save_values_codegen(a, b, c);

    std::cout << "g_acc0: " << g_acc0 << ", g_acc1: " << g_acc1 << ", g_signed: " << g_signed << std::endl;
    return 0;
}}
"""

def make_statement(ops, config):
    op = random.choice(ops)
    imm = config["rand_imm"]()
    sh = random.randint(1, 6)
    mask = random.choice([0xFF, 0xF, 0xAAAA_AAAA, 0x3333])
    return op.format(imm=imm, sh=sh, mask=mask)

def make_block(statements_per_block, instr_type):
    parts = []
    config = config_int if instr_type == "int" else config_float
    ops = integer_ops if instr_type == "int" else floating_ops

    for _ in range(statements_per_block):
        s = make_statement(ops, config)
        parts.append(f"\t{s};")
    # after the block, save values so it doesn't get optimised away
    parts.append("\tsave_values_codegen(x, y, z);")
    return "\n".join(parts)

def generate(blocks, statements_per_block, seed, instr_type):
    random.seed(seed)
    # create body by instantiating many template specializations in one function template
    # we build many blocks for the single template body using labels that refer to x,y,z
    body_lines = []

    config = config_int if instr_type == "int" else config_float
    main_type = config["main_type"]
    second_type = config["second_type"]
    main_init_literal = config["main_init_literal"]
    second_init_literal = config["second_init_literal"]
    l0 = config["main_random_literal"]()
    l1 = config["main_random_literal"]()
    l2 = config["second_random_literal"]()

    l3 = config["main_random_literal"]()
    l4 = config["main_random_literal"]()
    l5 = config["second_random_literal"]()

    # declare local volatile copies to encourage register spills and memory ops
    body_lines.append(f"\tvolatile {main_type} s = static_cast<{main_type}>(x + {l0});")
    body_lines.append(f"\tvolatile {main_type} t = static_cast<{main_type}>(y + {l1});")
    body_lines.append(f"\tvolatile {second_type} u = static_cast<{second_type}>(z - {l2});")

    for b in range(blocks):
        # insert a labelled mini-block to be invoked by template instantiation repetition
        block = make_block(statements_per_block, instr_type)
        body_lines.append(f"    // block {b}")
        body_lines.append(block)
    body = "\n".join(body_lines)
    return program_template.format(body=body, blocks=blocks, main_type=main_type, second_type=second_type, main_init_literal=main_init_literal, second_init_literal=second_init_literal, l0=l3, l1=l4, l2=l5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--blocks", type=int, default=20, help="Number of blocks")
    ap.add_argument("--statements-per-block", type=int, default=30, help="Statements per block")
    ap.add_argument("--out", type=str, default="random_ints_instrs.cpp")
    ap.add_argument("--out-dir", type=str, default="./rand_instructions")
    ap.add_argument("--instr-type", type=str, default="int", help="Either int or float")
    args = ap.parse_args()

    seed = args.seed
    code = generate(args.blocks, args.statements_per_block, seed, args.instr_type)
    
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.out

    out_file.write_text(code)
    print(f"Wrote {out_file} ({seed=} {args.blocks=} {args.statements_per_block=})")

if __name__ == "__main__":
    main()
