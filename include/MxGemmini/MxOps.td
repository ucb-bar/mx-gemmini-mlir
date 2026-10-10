include "mlir/IR/OpBase.td"

def MxGemmini_Dialect : Dialect {
  let name = "mx_gemmini";
  let cppNamespace = "::mlir::mx_gemmini";
  let summary = "Source-scoped MX Gemmini contract handoff";
}

class Mx_Op<string mnemonic> : Op<MxGemmini_Dialect, mnemonic> {
  let hasVerifier = 1;
}

def Mx_ResourceOp : Mx_Op<"resource"> {
  let summary = "A checked external MX operand, scale, or LUT resource";
  let arguments = (ins StrAttr:$site_id, StrAttr:$resource_name,
                   StrAttr:$resource_sha256, StrAttr:$resource_layout,
                   StrAttr:$payload_manifest_sha256,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$data);
}

def Mx_UploadLutOp : Mx_Op<"upload_lut"> {
  let summary = "Bind an indexed MX LUT bank before contraction";
  let arguments = (ins AnyTensor:$data, StrAttr:$site_id,
                   StrAttr:$lut_target, StrAttr:$payload_manifest_sha256,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}

def Mx_EncodeOp : Mx_Op<"encode"> {
  let summary = "Prepare packed operand codes and E8M0 scales";
  let arguments = (ins AnyTensor:$source, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$element_format,
                   OptionalAttr<StrAttr>:$projection,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_ContractOp : Mx_Op<"contract"> {
  let summary = "Accumulate an MX contraction in the selected mesh";
  let arguments = (ins AnyTensor:$lhs_codes, AnyTensor:$lhs_scales,
                   AnyTensor:$rhs_codes, AnyTensor:$rhs_scales,
                   StrAttr:$site_id, OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$activation_format,
                   OptionalAttr<StrAttr>:$weight_format,
                   OptionalAttr<StrAttr>:$activation_projection,
                   OptionalAttr<StrAttr>:$weight_projection,
                   OptionalAttr<I32Attr>:$pe_mode,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   OptionalAttr<StrAttr>:$payload_manifest_sha256,
                   OptionalAttr<StrAttr>:$payload_origin,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$accumulator);
}

def Mx_ReadoutBF16Op : Mx_Op<"readout_bf16"> {
  let summary = "Read BF16 from the accumulator";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$value);
}

def Mx_ReadoutQuantizedOp : Mx_Op<"readout_quantized"> {
  let summary = "Read packed MX output codes and E8M0 output scales";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   StrAttr:$output_format,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_ReadoutToSmemOp : Mx_Op<"readout_to_smem"> {
  let summary = "Write BF16 accumulator values to a shared-memory view";
  let arguments = (ins AnyTensor:$accumulator, AnyMemRef:$destination,
                   StrAttr:$site_id, OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_WaitOp : Mx_Op<"wait"> {
  let summary = "Wait for previously issued MX commands before a Muon consumer";
  let arguments = (ins StrAttr:$site_id, OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256,
                   StrAttr:$policy_sha256, StrAttr:$manifest_sha256);
}

def Mx_RequantizeOp : Mx_Op<"requantize"> {
  let summary = "Requantize accumulator output for an on-chip consumer";
  let arguments = (ins AnyTensor:$accumulator, StrAttr:$site_id,
                   OptionalAttr<StrAttr>:$format,
                   OptionalAttr<StrAttr>:$output_format,
                   OptionalAttr<StrAttr>:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
  let results = (outs AnyTensor:$codes, AnyTensor:$scales);
}

def Mx_VpuExecuteOp : Mx_Op<"vpu_execute"> {
  let summary = "Issue a source-bound BF16 scratchpad VPU command";
  let arguments = (ins StrAttr:$site_id, StrAttr:$kind,
                   I32Attr:$src1_row, I32Attr:$src2_row, I32Attr:$dst_row,
                   I32Attr:$rows, I32Attr:$reduction_length,
                   BoolAttr:$broadcast, I32Attr:$immediate_bf16,
                   OptionalAttr<I32Attr>:$second_dst_row,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}

def Mx_SpadRequantOp : Mx_Op<"spad_requant"> {
  let summary = "Requantize BF16 scratchpad rows into resident MX operand A";
  let arguments = (ins StrAttr:$site_id,
                   I32Attr:$source_row, I32Attr:$destination_row,
                   I32Attr:$m, I32Attr:$n, StrAttr:$output_format,
                   BoolAttr:$tiled, BoolAttr:$resident,
                   I64Attr:$scale_dram_address,
                   OptionalAttr<StrAttr>:$scale_buffer,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}

def Mx_ResidentContractOp : Mx_Op<"resident_contract"> {
  let summary = "Consume a resident MX activation tile and scales in a second contraction";
  let arguments = (ins StrAttr:$site_id,
                   I32Attr:$activation_row, I32Attr:$weight_row,
                   I32Attr:$output_row, I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   StrAttr:$activation_format, StrAttr:$weight_format,
                   StrAttr:$output_format,
                   StrAttr:$weight_buffer, StrAttr:$weight_scales_buffer,
                   StrAttr:$output_scales_buffer,
                   StrAttr:$profile_sha256,
                   StrAttr:$contract_sha256, StrAttr:$policy_sha256,
                   StrAttr:$manifest_sha256);
}
