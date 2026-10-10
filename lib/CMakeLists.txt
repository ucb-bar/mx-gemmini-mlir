add_mlir_dialect_library(MLIRMxGemmini
  MxDialect.cpp
  MxOps.cpp
  DEPENDS MLIRMxOpsIncGen
  LINK_LIBS PUBLIC MLIRIR LLVMSupport)
