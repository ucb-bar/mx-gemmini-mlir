#include "MxGemmini/MxOps.h"
#include "MxGemmini/MxDialect.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
using namespace mlir::mx_gemmini;

static LogicalResult verifyBinding(Operation *op, bool requiresFormat) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return op->emitOpError("requires a module");
  for (auto pair : {std::pair{"contract_sha256", "mx.contract_sha256"},
                    std::pair{"policy_sha256", "mx.policy_sha256"},
                    std::pair{"manifest_sha256", "prov.quantization_manifest_sha256"}}) {
    auto local = op->getAttrOfType<StringAttr>(pair.first);
    auto selected = module->getAttrOfType<StringAttr>(pair.second);
    if (!local || !selected || local.getValue() != selected.getValue() ||
        local.getValue().size() != 64)
      return op->emitOpError("digest differs from selected module binding: ") << pair.second;
  }
  auto site = op->getAttrOfType<StringAttr>("site_id");
  if (!site || site.getValue().empty()) return op->emitOpError("requires a site ID");
  if (requiresFormat) {
    auto format = op->getAttrOfType<StringAttr>("format");
    if (!format || (format.getValue() != "mxfp8" && format.getValue() != "mxfp6" &&
                    format.getValue() != "mxfp4"))
      return op->emitOpError("requires a selected MX format");
  }
  return success();
}

LogicalResult EncodeOp::verify() { return verifyBinding(*this, true); }
LogicalResult ContractOp::verify() { return verifyBinding(*this, true); }
LogicalResult ReadoutBF16Op::verify() { return verifyBinding(*this, false); }
LogicalResult RequantizeOp::verify() { return verifyBinding(*this, true); }

#define GET_OP_CLASSES
#include "MxGemmini/MxOps.cpp.inc"
