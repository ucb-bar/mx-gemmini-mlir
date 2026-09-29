add_executable(mx-gemmini-opt mx-gemmini-opt.cpp)
target_link_libraries(mx-gemmini-opt PRIVATE MLIRMxGemmini MLIROptLib MLIRIR MLIRFuncDialect)
mlir_check_all_link_libraries(mx-gemmini-opt)
