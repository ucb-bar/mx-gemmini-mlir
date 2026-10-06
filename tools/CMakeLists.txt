add_executable(mx-gemmini-opt mx-gemmini-opt.cpp)
target_link_libraries(mx-gemmini-opt PRIVATE MLIRMxGemmini MLIROptLib MLIRIR
  MLIRFuncDialect MLIRAffineDialect MLIRArithDialect MLIRControlFlowDialect
  MLIRLinalgDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
  MLIRTensorDialect MLIRVectorDialect)
mlir_check_all_link_libraries(mx-gemmini-opt)
