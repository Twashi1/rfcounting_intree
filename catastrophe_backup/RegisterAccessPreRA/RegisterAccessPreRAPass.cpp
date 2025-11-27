#include "llvm/CodeGen/RegisterAccessPreRAPass.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FileSystem.h"

#include "llvm/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <stack>
#include <unordered_map>
#include <unordered_set>
// TODO: use numeric limits instead
#include <cfloat>

#define DEBUG_TYPE "reg-access-prera"

using namespace llvm;

namespace llvm {
char RegisterAccessPreRAPass::ID = 0;
unsigned RegisterAccessPreRAPass::Processed = 0;
unsigned RegisterAccessPreRAPass::Total = 0;
ExtPathCollector RegisterAccessPreRAPass::PC = {};

void ExtPathCollector::buildCriticalPath() {
  // Build global adjacency list
  // - for each basic block, we need its personal list of successors
  // - for each machine function, we have its basic block, and all machine
  //    functions it links to
  // Find SCCs Topological sort Use SCCs to find critical
  // path Return critical path and print in nice way
}

void ExtPathCollector::addMachineFunctionEdge(const std::string &Caller,
                                              const std::string &Callee) {
  registerFunction(Caller);
  registerFunction(Callee);

  unsigned CallerID = FunctionIDs[Caller];
  unsigned CalleeID = FunctionIDs[Callee];

  FunctionMetadata[CallerID].Successors.push_back(CalleeID);
}

unsigned ExtPathCollector::registerFunction(const std::string &FunctionName) {
  if (!FunctionIDs.count(FunctionName)) {
    FunctionIDs[FunctionName] = FunctionIDCount++;
    ExtFunctionMetadata Metadata;
    Metadata.FunctionName = FunctionName;
    FunctionMetadata.push_back(Metadata);
  }

  return FunctionIDs[FunctionName];
}

unsigned ExtPathCollector::registerBasicBlock(const std::string &FunctionName,
                                              unsigned LocalBlockID) {
  registerFunction(FunctionName);

  uint64_t BlockUniqueIdentifier =
      getUniqueBlockIdentifier(FunctionName, LocalBlockID);

  if (!BlockIDs.count(BlockUniqueIdentifier)) {
    BlockIDs[BlockUniqueIdentifier] = BlockIDCount++;
    // TODO: push_back can probably auto-call initializer
    ExtBBStats Stats;
    Stats.FunctionName = FunctionName;
    BlockStats.push_back(Stats);
    GlobalAdjacencyList.push_back(std::vector<unsigned>({}));
  }

  return BlockIDs[BlockUniqueIdentifier];
}

void RegisterAccessPreRAPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  // TODO: might need to add required of children?
  // AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

ExtFunctionMetadata
RegisterAccessPreRAPass::getFunctionMetadata(const std::string &FunctionName) {
  registerFunction(FunctionName);
  unsigned FunctionID = FunctionIDs[FunctionName];

  return FunctionMetadata[FunctionID];
}
void RegisterAccessPreRAPass::setFunctionMetadata(
    const ExtFunctionMetadata &FunctionMetadata,
    const std::string &FunctionName) {
  registerFunction(FunctionName);
  unsigned FunctionID = FunctionIDs[FunctionName];

  FunctionMetadata[FunctionID] = FunctionMetadata;
}
uint64_t RegisterAccessPreRAPass::getUniqueBlockIdentifier(
    const std::string &FunctionName, unsigned LocalBlockID) {
  registerFunction(FunctionName);

  unsigned FunctionID = FunctionIDs[FunctionName];
  // misnomer, but can't think of a good name
  uint64_t BlockUniqueIdentifier = (static_cast<uint64_t>(FunctionID) << 32) |
                                   static_cast<uint32_t>(LocalBlockID);

  return BlockUniqueIdentifier;
}

ExtBBStats &RegisterAccessPreRAPass::getBBStats(const std::string &FunctionName,
                                                unsigned LocalBlockID) {
  registerBasicBlock(FunctionName, LocalBlockID);

  uint64_t BlockUniqueIdentifier =
      getUniqueBlockIdentifier(FunctionName, LocalBlockID);
  unsigned BlockID = BlockIDs[BlockUniqueIdentifier];

  return BlockStats[BlockID];
}

bool RegisterAccessPreRAPass::runOnMachineFunction(MachineFunction &MF) {
  if (!Total) {
    for (const Function &F : *MF.getFunction().getParent()) {
      if (!F.isDeclaration()) {
        Total++;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Found " << Total << " machine functions\n");

  LLVM_DEBUG(dbgs() << "Running RegisterAccessPreRAPass on " << MF.getName()
                    << "\n");

  const std::string MFName = MF.getName().str();

  PC.registerFunction(MFName);

  const TargetSubtargetInfo &TSI = MF.getSubtarget();
  // const TargetInstrInfo *TTI = TSI.getInstrInfo();
  TargetSchedModel SchedModel;
  SchedModel.init(&TSI);

  // TODO: can probably remove MBPI
  auto *MBPIWrapper =
      getAnalysisIfAvailable<MachineBranchProbabilityInfoWrapperPass>();
  MachineBlockFrequencyInfoWrapperPass *MBFIWrapper =
      getAnalysisIfAvailable<MachineBlockFrequencyInfoWrapperPass>();
  MachineBlockFrequencyInfo *MBFI = nullptr;

  if (MBFIWrapper == nullptr) {
    LLVM_DEBUG(dbgs() << "MBFI wrapper is nullptr\n");

    return false;
  } else {
    MBFI = &MBFIWrapper->getMBFI();

    LLVM_DEBUG(dbgs() << "MBFI wrapper is not nullptr\n");
  }

  if (MBPIWrapper == nullptr) {
    LLVM_DEBUG(dbgs() << "MBPI wrapper is nullptr\n");
  } else {
    LLVM_DEBUG(dbgs() << "MBPI wrapper is not nullptr\n");
  }

  std::error_code EC;
  raw_fd_ostream OutFile("reg_stats.csv", EC, sys::fs::OF_Append);

  if (EC) {
    errs() << "Error opening file: " << EC.message() << "\n";
    return false;
  }

  // assign local ID to each block
  // TODO: is Blocks ever used?
  std::vector<MachineBasicBlock *> Blocks;
  std::unordered_map<MachineBasicBlock *, unsigned> BlockIDs;
  Blocks.reserve(MF.size());

  unsigned BlockID = 0;
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs() << "Collecting info for MBB: " << MBB.getName() << "\n");

    // this is the entry block, record entry block ID for this machine function
    if (BlockID == 0) {
      ExtFunctionMetadata FunctionMetadata = PC.getFunctionMetadata(MFName);
      FunctionMetadata.EntryBasicBlock = PC.registerBasicBlock(MFName, BlockID);
      PC.setFunctionMetadata(FunctionMetadata, MFName);
    }

    Blocks.push_back(&MBB);
    BlockIDs.insert({&MBB, BlockID++});

    ExtBBStats &BlockStat = PC.getBBStats(MFName, BlockIDs[&MMB]);
    BlockStat.Cycles = 0.0;
    BlockStat.Freq = 1.0;
    BlockStat.GlobalFreq = 1.0;
    BlockStat.InstrCount = 0.0;
    BlockStat.Loads = 0.0;
    BlockStat.Stores = 0.0;
    BlockStat.Reads = 0.0;
    BlockStat.Writes = 0.0;
    BlockStat.Reloads = 0.0;
    BlockStat.Spills = 0.0;
    BlockStat.Name = MBB.getName().str();
    BlockStat.FunctionName = MFName;

    for (auto &MI : MBB) {
      if (MI.isDebugInstr() || MI.isPseudo()) {
        continue;
      }

      // instruction latency
      unsigned Latency = 1;
      BlockStat.InstrCount += 1.0;

      if (SchedModel.hasInstrSchedModel()) {
        Latency = SchedModel.computeInstrLatency(&MI);
      }

      BlockStat.Cycles += static_cast<double>(Latency);

      // TODO: num spills/reloads from frame index operands
      // number of stores/loads, so modelling cache hopefully
      if (MI.mayLoad()) {
        BlockStat.Loads += 1.0;
      }

      if (MI.mayStore()) {
        BlockStat.Stores += 1.0;
      }

      // TODO: remove?
      // ignore invalid operands
      if (MI.getOpcode() == TargetOpcode::STACKMAP ||
          MI.getOpcode() == TargetOpcode::PATCHPOINT) {
        continue;
      }

      for (unsigned i = 0; i < MI.getNumOperands(); i++) {
        const MachineOperand &MO = MI.getOperand(i);

        if (MI.isCall()) {
          const MachineOperand *Callee =
              MI.getOperand(0).isGlobal() ? &MI.getOperand(0) : nullptr;

          if (Callee) {
            const Function *F = dyn_cast<Function>(Callee->getGlobal());

            if (F) {
              std::string CalleeName = F->getName().str();
              std::string OurName = MF.getFunction().getName().str();

              PC.addMachineFunctionEdge(OurName, CalleeName);
            }
          }
        }

        // TODO: this might cover the above opcode conditions
        if (!MO.isReg()) {
          continue;
        }

        if (MO.isUse()) {
          BlockStat.Reads += 1.0;
        }

        if (MO.isDef()) {
          BlockStat.Writes += 1.0;
        }
      }
    }

    // if info available, get execution frequency
    if (MBFI != nullptr) {
      // TODO: += because sometimes result is 0.0, but still weird
      BlockStat.Freq += MBFI->getBlockFreqRelativeToEntryBlock(&MBB);
      BlockStat.GlobalFreq += MBFI->getBlockFreq(&MBB);
    }

    LLVM_DEBUG(dbgs() << "\tMBB Cycles " << BlockStat.Cycles << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Frequency " << BlockStat.Freq << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Global Frequency " << BlockStat.GlobalFreq
                      << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Loads per kilo cycles "
                      << BlockStat.Loads * 1000.0 / BlockStat.Cycles << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Stores per kilo cycles "
                      << BlockStat.Stores * 1000.0 / BlockStat.Cycles << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Register reads per kilo cycles "
                      << BlockStat.Reads * 1000.0 / BlockStat.Cycles << "\n");
    LLVM_DEBUG(dbgs() << "\tMBB Register writes per kilo cycles "
                      << BlockStat.Writes * 1000.0 / BlockStat.Cycles << "\n");
  }

  // make adjacency list
  // TODO: what if index fails
  std::vector<std::vector<unsigned>> AdjacencyList(Blocks.size());

  for (auto *MBB : Blocks) {
    unsigned u = BlockIDs[MBB];

    for (auto *Successor : MBB->successors()) {
      unsigned v = BlockIDs[Successor];
      AdjacencyList[u].push_back(v);
    }
  }

  // condense strongly connected components into one large node
  // using Tarjan's articulation points algorithm
  unsigned N = Blocks.size();
  std::vector<int> Index(N, -1);
  std::vector<int> LowLink(N, -1);
  std::vector<int> OnStack(N, 0);
  std::stack<unsigned> S;

  // component IDs
  std::vector<int> CompID(N, -1);
  unsigned IndexTarjan = 0;
  int CompCount = 0;

  // TODO: refactor to standalone function
  // can't use auto because of recursion
  std::function<void(unsigned)> StronglyConnect = [&](unsigned v) {
    Index[v] = IndexTarjan;
    LowLink[v] = IndexTarjan;
    IndexTarjan++;

    S.push(v);
    OnStack[v] = 1;

    for (unsigned w : AdjacencyList[v]) {
      if (Index[w] == -1) {
        StronglyConnect(w);
        LowLink[v] = std::min(LowLink[v], LowLink[w]);
      } else if (OnStack[w]) {
        LowLink[v] = std::min(LowLink[v], Index[w]);
      }
    }

    if (LowLink[v] == Index[v]) {
      // start new component
      // this condition shouldn't really matter
      //  but just for sanity, I don't want a while true
      while (!S.empty()) {
        unsigned w = S.top();
        S.pop();

        OnStack[w] = 0;
        CompID[w] = CompCount;

        if (w == v) {
          break;
        }
      }

      CompCount++;
    }
  };

  // create SCCs
  for (unsigned v = 0; v < N; v++) {
    if (Index[v] == -1) {
      StronglyConnect(v);
    }
  }

  // build DAG from SCCs
  std::vector<double> CompCost(CompCount, 0.0);
  std::vector<double> CompWeight(CompCount, 0.0);

  for (unsigned v = 0; v < N; v++) {
    int c = CompID[v];
    CompCost[c] += Stats[v].Cycles;
    CompWeight[c] += Stats[v].Cycles * Stats[v].Freq;
  }

  std::vector<std::vector<int>> DAGAdjacency =
      std::vector<std::vector<int>>(CompCount);
  // duplicate detection, two vertices packed
  std::unordered_set<uint64_t> DAGEdges;

  for (unsigned u = 0; u < N; u++) {
    for (unsigned v : AdjacencyList[u]) {
      int cu = CompID[u];
      int cv = CompID[v];

      if (cu != cv) {
        // maybe order them, but uv not same as vu?
        uint64_t key =
            (static_cast<uint64_t>(cu) << 32) | static_cast<uint32_t>(cv);

        // no contains in c++17 :(
        if (!DAGEdges.count(key)) {
          DAGAdjacency[cu].push_back(cv);
          DAGEdges.insert(key);
        }
      }
    }
  }

  // topological sort (Kahn's algorithm)
  // InDegree can be thought as number of unfulfilled dependencies
  std::vector<int> InDegree(CompCount, 0);

  for (int u = 0; u < CompCount; u++) {
    for (int v : DAGAdjacency[u]) {
      InDegree[v]++;
    }
  }

  std::vector<int> Topo;
  Topo.reserve(CompCount);
  std::deque<int> q;

  // get all starting nodes, ones with no dependencies
  for (int i = 0; i < CompCount; i++) {
    if (InDegree[i] == 0) {
      q.push_back(i);
    }
  }

  while (!q.empty()) {
    int u = q.front();
    q.pop_front();

    Topo.push_back(u);

    for (int v : DAGAdjacency[u]) {
      // if all dependencies fulfilled, we can schedule it
      if (--InDegree[v] == 0) {
        q.push_back(v);
      }
    }
  }

  // Topo has topologically sorted components
  // TODO: -DOUBLE_MAX? i mean good enough anyway
  const double NEG_INF = -FLT_MAX;

  std::vector<double> BestWeight = std::vector<double>(CompCount, NEG_INF);
  std::vector<int> Predecessor = std::vector<int>(CompCount, -1);

  for (int Node : Topo) {
    if (BestWeight[Node] == NEG_INF) {
      BestWeight[Node] = CompWeight[Node];
    }

    // relax
    for (int v : DAGAdjacency[Node]) {
      double Candidate = BestWeight[Node] + CompWeight[v];

      if (Candidate > BestWeight[v]) {
        BestWeight[v] = Candidate;
        Predecessor[v] = Node;
      }
    }
  }

  // find max-end node
  double BestOverall = NEG_INF;
  int BestNode = -1;

  for (int i = 0; i < CompCount; i++) {
    if (BestWeight[i] > BestOverall) {
      BestOverall = BestWeight[i];
      BestNode = i;
    }
  }

  // reconstruct path
  std::vector<int> CompPath;

  // a bit abusive of the for loop syntax but whatever
  for (int Cur = BestNode; Cur != -1; Cur = Predecessor[Cur]) {
    CompPath.push_back(Cur);
  }

  std::reverse(CompPath.begin(), CompPath.end());

  std::vector<std::vector<unsigned>> BlocksInComp =
      std::vector<std::vector<unsigned>>(CompCount);

  for (unsigned v = 0; v < N; v++) {
    BlocksInComp[CompID[v]].push_back(v);
  }

  // Write to CSV
  OutFile << "function,total_weight,component_path,blocks_in_component\n";
  OutFile << '"' << MF.getFunction().getName().str() << '"' << ','
          << BestOverall << ',' << '"';

  for (size_t i = 0; i < CompPath.size(); i++) {
    if (i)
      OutFile << "->";
    OutFile << CompPath[i];
  }

  OutFile << "\",\"";
  for (size_t i = 0; i < CompPath.size(); i++) {
    if (i)
      OutFile << ";";
    int c = CompPath[i];
    bool First = true;

    for (unsigned b : BlocksInComp[c]) {
      if (!First)
        OutFile << "|";
      OutFile << b;
      First = false;
    }
  }
  OutFile << "\"\n";
  OutFile.close();

  if (Processed == Total) {
    // TODO: perform critical path computation
    LLVM_DEBUG(dbgs() << "Perform critical path computation now...\n");

    PC.buildCriticalPath();
  }

  return false;
}
} // namespace llvm

INITIALIZE_PASS(RegisterAccessPreRAPass, "reg-access-prera",
                "Register Access Pre-RA Pass", false, false)
// INITIALIZE_PASS_END(RegisterAccessPreRAPass, "reg-access-prera",
//                    "Register Access Pre-RA Pass", false, false)

namespace llvm {
FunctionPass *createRegisterAccessPreRAPass() {
  return new RegisterAccessPreRAPass();
}

#undef DEBUG_TYPE

} // namespace llvm
