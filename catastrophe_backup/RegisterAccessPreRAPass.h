#ifndef LLVM_CODEGEN_REGISTERACCESSPRERAPASS_H
#define LLVM_CODEGEN_REGISTERACCESSPRERAPASS_H

#include "llvm/CodeGen/MachineFunctionPass.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {

// For each BB we need its individual stats
// for each function we need its links to other functions

struct ExtBBStats {
  double Cycles;
  double Freq;
  double GlobalFreq;
  double Loads;
  double Stores;
  double Spills;
  double Reloads;
  double Reads;
  double Writes;
  double InstrCount;
  std::string Name;
  std::string FunctionName;
};

struct ExtFunctionMetadata {
  // FunctionIDs of successors
  std::vector<unsigned> Successors;
  unsigned EntryBasicBlock;
  std::string FunctionName;
};

struct ExtPathCollector {
  std::unordered_map<std::string, unsigned> FunctionIDs;
  std::unordered_map<uint64_t, unsigned> BlockIDs;
  std::vector<ExtBBStats> BlockStats;
  std::vector<ExtFunctionMetadata> FunctionMetadata;
  std::vector<std::vector<unsigned>> GlobalAdjacencyList;

  unsigned BlockIDCount = 0;
  unsigned FunctionIDCount = 0;

  void addMachineFunctionEdge(const std::string &Caller,
                              const std::string &Callee);
  unsigned registerFunction(const std::string &FunctionName);
  unsigned registerBasicBlock(const std::string &FunctionName,
                              unsigned LocalBlockID);
  uint64_t getUniqueBlockIdentifier(const std::string &FunctionName,
                                    unsigned LocalBlockID);
  ExtBBStats &getBBStats(const std::string &FunctionName,
                         unsigned LocalBlockID);
  // TODO: note its a much more likely access pattern that you would change the
  // FunctionMetadata vector while preserving a reference to function metadata
  // thus we must have this getter/setter stuff
  ExtFunctionMetadata getFunctionMetadata(const std::string &FunctionName);
  void setFunctionMetadata(const ExtFunctionMetadata &FunctionMetadata,
                           const std::string &FunctionName);
  void buildCriticalPath();
};

class RegisterAccessPreRAPass : public MachineFunctionPass {
public:
  static char ID;
  static unsigned Processed;
  static unsigned Total;
  static ExtPathCollector PC;
  RegisterAccessPreRAPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "Register Access Pre-RA Pass";
  }
};

FunctionPass *createRegisterAccessPreRAPass();

} // namespace llvm

#endif // LLVM_CODEGEN_REGISTERACCESSPRERAPASS_H
