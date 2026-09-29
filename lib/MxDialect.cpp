#include "MxGemmini/MxDialect.h"
#include "MxGemmini/MxOps.h"
using namespace mlir;
using namespace mlir::mx_gemmini;
#include "MxGemmini/MxOpsDialect.cpp.inc"
void MxGemminiDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MxGemmini/MxOps.cpp.inc"
      >();
}
