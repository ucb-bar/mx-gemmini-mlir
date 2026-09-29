#include "MxGemmini/MxDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mx_gemmini::MxGemminiDialect, mlir::func::FuncDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
                                                  "MX Gemmini contract handoff\n", registry));
}
