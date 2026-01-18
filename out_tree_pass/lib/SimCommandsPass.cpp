#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"

#define DEBUG_TYPE "sim-commands-pass"

using namespace llvm;

const unsigned long MHZ_DEFAULT = 3400;

namespace {

struct AddSimCommandsPassOld : public FunctionPass {
  static char ID;
  AddSimCommandsPassOld() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    Module *M = F.getParent();
    LLVMContext &C = M->getContext();

    FunctionCallee SimCmd =
      M->getOrInsertFunction("set_sim_cmd",
          FunctionType::get(Type::getVoidTy(C), false));

    bool Changed = false;
    for (BasicBlock &BB : F) {
      IRBuilder<> B(&*BB.getFirstInsertionPt());
      B.CreateCall(SimCmd);
      Changed = true;
    }
    return Changed;
  }
};

} // end anonymous namespace

char AddSimCommandsPassOld::ID = 0;

extern "C" ::llvm::Pass *createAddSimCommandsPass() {
  return new AddSimCommandsPassOld();
}

static RegisterPass<AddSimCommandsPassOld>
    X("sim-commands-pass", "Add SIM Commands Pass", false, false);
