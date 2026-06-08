"""

NOTE: Script just for information on what the cache hiearchy was during runs
- we don't apply DVS in this script according to any marker/DVS calling point

scons build/ALL/gem5.opt
./build/ALL/gem5.opt \
    configs/custom.py

./build/X86/gem5.fast configs/custom.py

# Note: This config script will only run on an X86 host.
# Each run takes ~2 hours

"""

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import BinaryResource
import argparse

# TODO: in order to be more in line with the original paper, set L2 size to a MB, ensure these values are being read as kilobytes, not kilibits; seems like its converting to KiB

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="64kB",
    l1i_size="64kB",
    l2_size="512kB",
)

memory = SingleChannelDDR4_2400("4GB")

processor = SimpleProcessor(cpu_type=CPUTypes.O3, num_cores=1, isa=ISA.X86)

# TODO: 2h @ 3GHz, does changing clock frequency affect the cache misses/hits significantly?
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
args = parser.parse_args()

# TODO: set our own workload
board.set_se_binary_workload(binary=BinaryResource(local_path=args.binary))

simulator = Simulator(
    board=board,
)

simulator.run()
