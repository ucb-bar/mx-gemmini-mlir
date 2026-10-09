#include "MxGemmini/MxOps.h"
#include "MxGemmini/MxDialect.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
using namespace mlir::mx_gemmini;

static bool isLegacyFormat(StringRef value) {
  return value == "mxfp8" || value == "mxfp6" || value == "mxfp4";
}

static bool isNamedFormat(StringRef value) {
  return value == "fp4_e2m1" || value == "fp6_e2m3" ||
         value == "fp6_e3m2" || value == "fp8_e4m3" ||
         value == "fp8_e5m2";
}

static bool isProjection(StringRef value) {
  return value == "direct" || value == "lut";
}

static LogicalResult verifyBinding(Operation *op) {
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
  auto localProfile = op->getAttrOfType<StringAttr>("profile_sha256");
  auto moduleProfile = module->getAttrOfType<StringAttr>("mx.profile_sha256");
  if (localProfile || moduleProfile) {
    if (!localProfile || !moduleProfile ||
        localProfile.getValue().size() != 64 ||
        localProfile.getValue() != moduleProfile.getValue())
      return op->emitOpError("profile digest differs from mx.profile_sha256");
  }
  return success();
}

LogicalResult EncodeOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto legacy = (*this)->getAttrOfType<StringAttr>("format");
  auto named = (*this)->getAttrOfType<StringAttr>("element_format");
  auto projection = (*this)->getAttrOfType<StringAttr>("projection");
  if (legacy && !named && !projection && isLegacyFormat(legacy.getValue()))
    return success();
  if (!legacy && named && isNamedFormat(named.getValue()) && projection &&
      isProjection(projection.getValue()) &&
      (*this)->hasAttr("profile_sha256"))
    return success();
  return emitOpError("requires either a legacy MX format or a profile-bound named element format and projection");
}

LogicalResult ContractOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto modulePayload = module->getAttrOfType<StringAttr>("mx.payload_manifest_sha256");
  auto localPayload = (*this)->getAttrOfType<StringAttr>("payload_manifest_sha256");
  auto origin = (*this)->getAttrOfType<StringAttr>("payload_origin");
  if (modulePayload || localPayload || origin) {
    if (!modulePayload || !localPayload || !origin ||
        modulePayload.getValue() != localPayload.getValue() ||
        localPayload.getValue().size() != 64 ||
        origin.getValue() != "radiance_source_header_specialization")
      return emitOpError("source payload differs from module binding");
  }
  auto legacy = (*this)->getAttrOfType<StringAttr>("format");
  auto act = (*this)->getAttrOfType<StringAttr>("activation_format");
  auto weight = (*this)->getAttrOfType<StringAttr>("weight_format");
  auto actProjection = (*this)->getAttrOfType<StringAttr>("activation_projection");
  auto weightProjection = (*this)->getAttrOfType<StringAttr>("weight_projection");
  auto mode = (*this)->getAttrOfType<IntegerAttr>("pe_mode");
  if (legacy && !act && !weight && !actProjection && !weightProjection &&
      !mode && isLegacyFormat(legacy.getValue()))
    return success();
  if (!legacy && act && weight && actProjection && weightProjection &&
      isNamedFormat(act.getValue()) && isNamedFormat(weight.getValue()) &&
      isProjection(actProjection.getValue()) &&
      isProjection(weightProjection.getValue()) &&
      (!mode || (mode.getInt() >= 0 && mode.getInt() <= 11)) &&
      (*this)->hasAttr("profile_sha256"))
    return success();
  return emitOpError("requires either a legacy MX format or profile-bound activation/weight formats and projections");
}

LogicalResult ReadoutBF16Op::verify() { return verifyBinding(*this); }
LogicalResult ReadoutQuantizedOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto output = (*this)->getAttrOfType<StringAttr>("output_format");
  if (!output || !isNamedFormat(output.getValue()) ||
      !(*this)->hasAttr("profile_sha256"))
    return emitOpError("requires a profile-bound named output format");
  auto codes = dyn_cast<RankedTensorType>(getCodes().getType());
  auto scales = dyn_cast<RankedTensorType>(getScales().getType());
  if (!codes || !scales || codes.getRank() != 2 || scales.getRank() != 2 ||
      !codes.getElementType().isInteger(8) ||
      !scales.getElementType().isInteger(8))
    return emitOpError("requires rank-two i8 code and scale tensors");
  if (codes.hasStaticShape() && scales.hasStaticShape() &&
      (((output.getValue() == "fp4_e2m1" ||
         output.getValue() == "fp6_e3m2" || output.getValue() == "fp6_e2m3")
            ? codes.getDimSize(0) * 2 != scales.getDimSize(0)
            : codes.getDimSize(0) != scales.getDimSize(0)) ||
       codes.getDimSize(1) % 32 != 0 ||
       codes.getDimSize(1) / 32 != scales.getDimSize(1)))
    return emitOpError("code and E8M0 scale tensor shapes differ");
  return success();
}
LogicalResult ReadoutToSmemOp::verify() { return verifyBinding(*this); }
LogicalResult WaitOp::verify() { return verifyBinding(*this); }

LogicalResult RequantizeOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto legacy = (*this)->getAttrOfType<StringAttr>("format");
  auto output = (*this)->getAttrOfType<StringAttr>("output_format");
  if (legacy && !output && isLegacyFormat(legacy.getValue())) return success();
  if (!legacy && output && isNamedFormat(output.getValue()) &&
      (*this)->hasAttr("profile_sha256"))
    return success();
  return emitOpError("requires either a legacy MX format or a profile-bound named output format");
}

LogicalResult VpuExecuteOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto kind = (*this)->getAttrOfType<StringAttr>("kind").getValue();
  if (kind != "add" && kind != "sub" && kind != "mul" && kind != "adds" &&
      kind != "muls" && kind != "exp" && kind != "rcp" && kind != "rsqrt" &&
      kind != "rmax" && kind != "rsum" && kind != "ramax" && kind != "max" &&
      kind != "expsub" && kind != "expsum")
    return emitOpError("has an unknown VPU kind");
  auto row = [this](StringRef name) { return (*this)->getAttrOfType<IntegerAttr>(name).getInt(); };
  if (row("src1_row") < 0 || row("src1_row") > 0x3fff ||
      row("src2_row") < 0 || row("src2_row") > 0x3fff ||
      row("dst_row") < 0 || row("dst_row") > 0x3fff ||
      row("rows") <= 0 || row("rows") > 0xffff ||
      row("reduction_length") <= 0 || row("reduction_length") > 1023 ||
      row("immediate_bf16") < 0 || row("immediate_bf16") > 0xffff)
    return emitOpError("VPU address, row count, reduction length, or immediate is out of range");
  auto second = (*this)->getAttrOfType<IntegerAttr>("second_dst_row");
  if ((kind == "expsum") != static_cast<bool>(second) ||
      (second && (second.getInt() < 0 || second.getInt() > 0x3fff)))
    return emitOpError("EXPSUM alone requires a 14-bit second destination");
  if ((kind == "rmax" || kind == "rsum" || kind == "ramax" || kind == "expsum") &&
      row("rows") % row("reduction_length") != 0)
    return emitOpError("logical rows must divide the VPU input row count");
  return success();
}

LogicalResult SpadRequantOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto row = [this](StringRef name) { return (*this)->getAttrOfType<IntegerAttr>(name).getInt(); };
  auto output = (*this)->getAttrOfType<StringAttr>("output_format").getValue();
  if (output != "fp8_e4m3" && output != "fp4_e2m1")
    return emitOpError("SPAD_REQUANT output must be E4M3 or FP4");
  if (row("source_row") < 0 || row("source_row") > 0x3fff ||
      row("destination_row") < 0 || row("destination_row") > 0x3fff ||
      row("m") <= 0 || row("m") > 0xffff || row("n") <= 0 || row("n") > 0xffff)
    return emitOpError("SPAD_REQUANT address or dimension is out of range");
  int64_t blocks = row("m") * row("n") / 32;
  if (row("m") % 8 || row("n") % 32 || blocks % 32 || blocks > 2048)
    return emitOpError("SPAD_REQUANT requires M multiple of 8, N multiple of 32, and 32..2048 blocks");
  auto scale = row("scale_dram_address");
  if (scale < 0 || scale >= (int64_t{1} << 33))
    return emitOpError("SPAD_REQUANT scale address must fit 33 bits");
  if (auto buffer = (*this)->getAttrOfType<StringAttr>("scale_buffer")) {
    StringRef name = buffer.getValue();
    if (scale != 0 || name.empty())
      return emitOpError("SPAD_REQUANT scale buffer requires zero fixed address and a name");
    for (size_t i = 0; i < name.size(); ++i) {
      char c = name[i];
      bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
      bool digit = c >= '0' && c <= '9';
      if (!alpha && (i == 0 || !digit))
        return emitOpError("SPAD_REQUANT scale buffer name must be a C identifier");
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "MxGemmini/MxOps.cpp.inc"
