#include "MxGemmini/MxOps.h"
#include "MxGemmini/MxDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
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

static bool isSha256(StringRef value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char digit) {
           return (digit >= '0' && digit <= '9') ||
                  (digit >= 'a' && digit <= 'f');
         });
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

static bool isCheckedResource(Value value, StringRef expectedName,
                              StringRef payloadDigest) {
  auto resource = value.getDefiningOp<ResourceOp>();
  if (!resource) return false;
  auto name = resource->getAttrOfType<StringAttr>("resource_name");
  auto payload = resource->getAttrOfType<StringAttr>("payload_manifest_sha256");
  return name && payload && name.getValue() == expectedName &&
         payload.getValue() == payloadDigest;
}

LogicalResult ResourceOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto modulePayload = module->getAttrOfType<StringAttr>("mx.payload_manifest_sha256");
  auto localPayload = (*this)->getAttrOfType<StringAttr>("payload_manifest_sha256");
  auto name = (*this)->getAttrOfType<StringAttr>("resource_name");
  auto hash = (*this)->getAttrOfType<StringAttr>("resource_sha256");
  auto layout = (*this)->getAttrOfType<StringAttr>("resource_layout");
  if (!modulePayload || !localPayload || modulePayload != localPayload ||
      !name || !hash || !isSha256(hash.getValue()) || !layout ||
      layout.getValue().empty())
    return emitOpError("requires a checked module payload and resource descriptor");
  auto tensor = dyn_cast<RankedTensorType>(getData().getType());
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() != 2 ||
      (!tensor.getElementType().isInteger(8) &&
       !tensor.getElementType().isInteger(16) &&
       !tensor.getElementType().isInteger(32)) ||
      tensor.getDimSize(0) <= 0 || tensor.getDimSize(1) <= 0)
    return emitOpError("requires a static rank-two byte, halfword, or word tensor");
  return success();
}

LogicalResult UploadLutOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto target = (*this)->getAttrOfType<StringAttr>("lut_target");
  auto payload = (*this)->getAttrOfType<StringAttr>("payload_manifest_sha256");
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!target || !payload ||
      (target.getValue() != "activation" && target.getValue() != "weight" &&
       target.getValue() != "output") ||
      payload != module->getAttrOfType<StringAttr>("mx.payload_manifest_sha256") ||
      !isCheckedResource(getData(),
                         (target.getValue() + "_lut").str(), payload.getValue()))
    return emitOpError("requires a matching checked LUT resource");
  auto tensor = dyn_cast<RankedTensorType>(getData().getType());
  if (!tensor || tensor.getRank() != 2 || !tensor.hasStaticShape() ||
      !tensor.getElementType().isInteger(32))
    return emitOpError("requires a static rank-two i32 LUT bank");
  return success();
}

LogicalResult RuntimeLutOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto runtime = module->getAttrOfType<StringAttr>("mx.runtime_resources_sha256");
  auto source = module->getAttrOfType<StringAttr>("mx.payload_binding_schema");
  auto target = (*this)->getAttrOfType<StringAttr>("lut_target");
  auto buffer = (*this)->getAttrOfType<StringAttr>("runtime_buffer");
  auto groups = (*this)->getAttrOfType<IntegerAttr>("groups");
  auto bits = (*this)->getAttrOfType<IntegerAttr>("entry_bits");
  if (!runtime || !isSha256(runtime.getValue()) || source || !target ||
      (target.getValue() != "activation" && target.getValue() != "weight" &&
       target.getValue() != "output") || !buffer || buffer.getValue().empty() ||
      !groups || groups.getInt() < 1 || groups.getInt() > 64 ||
      !bits || bits.getInt() != 6)
    return emitOpError("requires a runtime-bound FP6 LUT bank");
  StringRef name = buffer.getValue();
  for (size_t i = 0; i < name.size(); ++i) {
    char c = name[i];
    bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    bool digit = (c >= '0' && c <= '9');
    if (!alpha && (i == 0 || !digit))
      return emitOpError("runtime LUT buffer name must be a C identifier");
  }
  auto tensor = dyn_cast<RankedTensorType>(getData().getType());
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() != 2 ||
      !tensor.getElementType().isInteger(32) ||
      tensor.getDimSize(0) != groups.getInt() || tensor.getDimSize(1) != 3)
    return emitOpError("requires a groups-by-three i32 LUT tensor");
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
  auto payloadJson = module->getAttrOfType<StringAttr>("mx.payload_manifest_json");
  auto localPayload = (*this)->getAttrOfType<StringAttr>("payload_manifest_sha256");
  auto origin = (*this)->getAttrOfType<StringAttr>("payload_origin");
  if (payloadJson && (!modulePayload ||
      llvm::toHex(llvm::SHA256::hash(
          llvm::arrayRefFromStringRef(payloadJson.getValue())), true) !=
          modulePayload.getValue()))
    return emitOpError("source resource manifest digest differs from module binding");
  if (modulePayload || localPayload || origin) {
    if (!modulePayload || !localPayload || !origin ||
        modulePayload.getValue() != localPayload.getValue() ||
        localPayload.getValue().size() != 64 ||
        (origin.getValue() != "radiance_source_header_specialization" &&
         origin.getValue() != "radiance_source_derived_attention_qk_candidate" &&
         origin.getValue() != "radiance_source_derived_attention_pv_proxy" &&
         origin.getValue() != "radiance_source_derived_gemm_fixture" &&
         origin.getValue() != "nicolas_source_header_specialization" &&
         origin.getValue() != "nicolas_generated_header_specialization"))
      return emitOpError("source payload differs from module binding");
  }
  auto resourceSchema = module->getAttrOfType<StringAttr>("mx.payload_binding_schema");
  if (resourceSchema) {
    if (resourceSchema.getValue() != "source_resources_ssa_v1" || !localPayload ||
        !isCheckedResource(getLhsCodes(), "activation", localPayload.getValue()) ||
        !isCheckedResource(getLhsScales(), "activation_scales", localPayload.getValue()) ||
        !isCheckedResource(getRhsCodes(), "weight", localPayload.getValue()) ||
        !isCheckedResource(getRhsScales(), "weight_scales", localPayload.getValue()))
      return emitOpError("contraction operands differ from source resource binding");
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

LogicalResult ReadoutBF16Op::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto layout = (*this)->getAttrOfType<StringAttr>("memory_layout");
  if (layout && layout.getValue() != "row_major_bf16" &&
      layout.getValue() != "output_tile_major_bf16")
    return emitOpError("requires row_major_bf16 or output_tile_major_bf16 memory layout");
  return success();
}
LogicalResult ReadoutQuantizedOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto output = (*this)->getAttrOfType<StringAttr>("output_format");
  auto projection = (*this)->getAttrOfType<StringAttr>("output_projection");
  if (!output || !isNamedFormat(output.getValue()) ||
      !(*this)->hasAttr("profile_sha256"))
    return emitOpError("requires a profile-bound named output format");
  if (projection && (projection.getValue() != "lut" ||
                     output.getValue() != "fp8_e4m3"))
    return emitOpError("supports LUT output projection only for E4M3");
  auto codes = dyn_cast<RankedTensorType>(getCodes().getType());
  auto scales = dyn_cast<RankedTensorType>(getScales().getType());
  if (!codes || !scales || codes.getRank() != 2 || scales.getRank() != 2 ||
      !codes.getElementType().isInteger(8) ||
      !scales.getElementType().isInteger(8))
    return emitOpError("requires rank-two i8 code and scale tensors");
  if (codes.hasStaticShape() && scales.hasStaticShape() &&
      ((((projection && projection.getValue() == "lut") ||
         output.getValue() == "fp4_e2m1" ||
         output.getValue() == "fp6_e3m2" || output.getValue() == "fp6_e2m3")
            ? codes.getDimSize(0) * 2 != scales.getDimSize(0)
            : codes.getDimSize(0) != scales.getDimSize(0)) ||
       codes.getDimSize(1) % 32 != 0 ||
       codes.getDimSize(1) / 32 != scales.getDimSize(1)))
    return emitOpError("code and E8M0 scale tensor shapes differ");
  return success();
}
LogicalResult HostRequantizeOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  auto input = dyn_cast<RankedTensorType>(getValue().getType());
  auto codes = dyn_cast<RankedTensorType>(getCodes().getType());
  auto scales = dyn_cast<RankedTensorType>(getScales().getType());
  auto output = (*this)->getAttrOfType<StringAttr>("output_format");
  auto policy = (*this)->getAttrOfType<StringAttr>("quant_policy");
  bool fp8 = output && policy && output.getValue() == "fp8_e4m3" &&
             policy.getValue() == "radiance_header_fp8_v1";
  bool fp6 = output && policy && output.getValue() == "fp6_e3m2" &&
             policy.getValue() == "radiance_header_fp6_lut_v1";
  if (!fp8 && !fp6)
    return emitOpError("requires an explicit Radiance FP8 or FP6 header policy");
  auto payload = (*this)->getParentOfType<ModuleOp>()->getAttrOfType<StringAttr>(
      "mx.payload_manifest_sha256");
  if ((fp8 && getOutputLut()) ||
      (fp6 && (!payload || !getOutputLut() || !isCheckedResource(
          getOutputLut(), "output_lut", payload.getValue()))))
    return emitOpError("output LUT differs from selected FP6 source resource");
  if (!input || input.getRank() != 2 || !input.getElementType().isBF16() ||
      !codes || codes.getRank() != 2 || !codes.getElementType().isInteger(8) ||
      !scales || scales.getRank() != 2 || !scales.getElementType().isInteger(8) ||
      !codes.hasStaticShape() || !scales.hasStaticShape() ||
      codes.getDimSize(0) * (fp6 ? 2 : 1) != scales.getDimSize(0) ||
      codes.getDimSize(1) != scales.getDimSize(1) * 32)
    return emitOpError("requires BF16 input and matching rank-two code/E8M0 scale tensors");
  if (input.hasStaticShape() &&
      (input.getDimSize(0) != codes.getDimSize(0) * (fp6 ? 2 : 1) ||
       input.getDimSize(1) != codes.getDimSize(1)))
    return emitOpError("BF16 input shape differs from output code shape");
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
  if (!(getInputs().empty() && getOutputs().empty()) &&
      !(getInputs().size() == 1 && getOutputs().size() == 1))
    return emitOpError("requires either no SSA edges or one BF16 input and output");
  if (!getInputs().empty()) {
    auto input = dyn_cast<RankedTensorType>(getInputs()[0].getType());
    auto output = dyn_cast<RankedTensorType>(getOutputs()[0].getType());
    if (!input || !output || input.getRank() != 2 ||
        !input.hasStaticShape() || !input.getElementType().isBF16() ||
        !output.getElementType().isBF16() || input != output)
      return emitOpError("connected VPU needs matching rank-two BF16 tensors");
  }
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
  if (!(getInputs().empty() && getOutputs().empty()) &&
      !(getInputs().size() == 1 && getOutputs().size() == 2))
    return emitOpError("requires either no SSA edges or BF16 input and code/scale outputs");
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
  if (!getInputs().empty()) {
    auto input = dyn_cast<RankedTensorType>(getInputs()[0].getType());
    auto codes = dyn_cast<RankedTensorType>(getOutputs()[0].getType());
    auto scales = dyn_cast<RankedTensorType>(getOutputs()[1].getType());
    int64_t m = row("m"), n = row("n");
    int64_t codeRows = output == "fp4_e2m1" ? m / 2 : m;
    if (!input || !input.getElementType().isBF16() ||
        !codes || !codes.getElementType().isInteger(8) ||
        !scales || !scales.getElementType().isInteger(8) ||
        input.getShape() != ArrayRef<int64_t>({m, n}) ||
        codes.getShape() != ArrayRef<int64_t>({codeRows, n}) ||
        scales.getShape() != ArrayRef<int64_t>({m, n / 32}))
      return emitOpError("connected SPAD_REQUANT tensor shapes differ from command geometry");
  }
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

LogicalResult ResidentContractOp::verify() {
  if (failed(verifyBinding(*this))) return failure();
  if (!(getInputs().empty() && getOutputs().empty()) &&
      !(getInputs().size() == 4 && getOutputs().size() == 2))
    return emitOpError("requires either no SSA edges or four inputs and two outputs");
  auto row = [this](StringRef name) { return (*this)->getAttrOfType<IntegerAttr>(name).getInt(); };
  if (row("activation_row") < 0 || row("activation_row") > 0x3fff ||
      row("weight_row") < 0 || row("weight_row") > 0x3fff ||
      row("output_row") < 0 || row("output_row") > 0x3fff ||
      row("m") <= 0 || row("n") <= 0 || row("k") <= 0 ||
      row("m") % 16 || row("n") % 16 || row("k") % 16)
    return emitOpError("resident contraction needs 14-bit rows and complete DIM16 tiles");
  auto format = [this](StringRef name) {
    return (*this)->getAttrOfType<StringAttr>(name).getValue();
  };
  StringRef precision = format("activation_format");
  if ((precision != "fp8_e4m3" && precision != "fp4_e2m1" &&
       precision != "fp6_e3m2") ||
      format("weight_format") != precision ||
      format("output_format") != precision)
    return emitOpError("resident contraction requires matching E4M3, E2M1, or E3M2 inputs and output");
  if (!getInputs().empty()) {
    int64_t m = row("m"), n = row("n"), k = row("k");
    int64_t pack = precision == "fp8_e4m3" ? 1 : 2;
    const int64_t expected[6][2] = {
        {m / pack, k}, {m, k / 32}, {k, n / pack}, {k / 32, n},
        {m / pack, n}, {m, n / 32}};
    for (unsigned i = 0; i < 6; ++i) {
      Value value = i < 4 ? getInputs()[i] : getOutputs()[i - 4];
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (!tensor || !tensor.getElementType().isInteger(8) ||
          tensor.getRank() != 2 || tensor.getDimSize(0) != expected[i][0] ||
          tensor.getDimSize(1) != expected[i][1])
        return emitOpError("connected resident contraction tensor shapes differ from command geometry");
    }
  }
  for (StringRef attr : {"weight_buffer", "weight_scales_buffer", "output_scales_buffer"}) {
    StringRef name = format(attr);
    if (name.empty()) return emitOpError("resident contraction buffer name is empty");
    for (size_t i = 0; i < name.size(); ++i) {
      char c = name[i];
      bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
      bool digit = (c >= '0' && c <= '9');
      if (!alpha && (i == 0 || !digit))
        return emitOpError("resident contraction buffer name must be a C identifier");
    }
  }
  if (precision == "fp6_e3m2") {
    auto groups = (*this)->getAttrOfType<IntegerAttr>("lut_groups");
    if (!groups || groups.getInt() != row("m") / 2 ||
        row("m") != row("n") || row("n") != row("k"))
      return emitOpError("FP6 resident contraction needs source-aligned LUT groups");
    for (StringRef attr : {"weight_lut_buffer", "activation_lut_buffer",
                           "output_lut_buffer"}) {
      auto buffer = (*this)->getAttrOfType<StringAttr>(attr);
      if (!buffer || buffer.getValue().empty())
        return emitOpError("FP6 resident contraction needs named LUT buffers");
      StringRef name = buffer.getValue();
      for (size_t i = 0; i < name.size(); ++i) {
        char c = name[i];
        bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        bool digit = (c >= '0' && c <= '9');
        if (!alpha && (i == 0 || !digit))
          return emitOpError("FP6 LUT buffer name must be a C identifier");
      }
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "MxGemmini/MxOps.cpp.inc"
