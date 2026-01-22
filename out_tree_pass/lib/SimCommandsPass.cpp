#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE \
  "sim-commands-pass"  // TODO: seems to be unused? or debug not working anyway

using namespace llvm;

const unsigned long MHZ_DEFAULT = 3400;

namespace {

struct AddSimCommandsPass : PassInfoMixin<AddSimCommandsPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& Manager) {
    errs() << "Running on function: " << F.getName() << "\n";

    Module* M = F.getParent();
    LLVMContext& C = M->getContext();

    FunctionCallee SimCmd = M->getOrInsertFunction(
        "ext_set_sim_cmd", FunctionType::get(Type::getVoidTy(C), false));
    FunctionCallee RoiStart = M->getOrInsertFunction(
        "ext_set_roi_start", FunctionType::get(Type::getVoidTy(C), false));
    FunctionCallee RoiEnd = M->getOrInsertFunction(
        "ext_set_roi_end", FunctionType::get(Type::getVoidTy(C), false));

    for (BasicBlock& BB : F) {
      IRBuilder<> BStart(&*BB.getFirstInsertionPt());
      CallInst* StartCall = BStart.CreateCall(RoiStart);

      IRBuilder<> BSim(StartCall->getNextNode());
      BSim.CreateCall(SimCmd);

      Instruction* Term = BB.getTerminator();
      IRBuilder<> BEnd(Term);
      BEnd.CreateCall(RoiEnd);
    }
    return PreservedAnalyses::none();
  }
};

}  // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AddSimCommandsPass", LLVM_VERSION_STRING,
          [](PassBuilder& PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager& FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sim-commands-pass") {
                    FPM.addPass(AddSimCommandsPass());
                    return true;
                  }
                  return false;
                });
          }};
}
